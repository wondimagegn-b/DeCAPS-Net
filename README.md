# DeCAPS-Net

Official code for **"Multimodal Human Parsing and Deformable Skeleton Fusion for Autism Spectrum Disorder Assessment"** (under review at *Pattern Recognition*).


## Installation

```bash
git clone https://github.com/wondimagegn-b/DeCAPS-Net.git
cd DeCAPS-Net

# 1) Install torch + torchvision matching your CUDA version from https://pytorch.org
#    (validated with torch 2.4.1 + torchvision 0.19.1, CUDA 12.1)

# 2) Install the rest
pip install -r requirements.txt
```

## Data

Download the datasets from their sources (this repo does not redistribute data):

- **GFBMD**: https://datadryad.org/dataset/doi:10.5061/dryad.s7h44j150
- **ASDPose**: https://github.com/Dinstein-Lab/ASDMotion (download `Annotated Dataset for Training.pkl` from the link given there)
- **Sapiens segmentation checkpoint** (needed for GFBMD parsing maps):
  https://huggingface.co/facebook/sapiens-seg-1b-torchscript → `sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2`

Expected raw GFBMD layout:

```text
<gfbmd_root>/
├── Autism/children with ASD/<id>/video/video.avi
├── Autism/children with ASD/<id>/video/8.xlsx
└── Typical/<id>/video/{video.avi, 8.xlsx}
```

After preprocessing (below), the repo expects:

```text
data/
├── asdpose/
│   └── ASDpose.h5
└── gfbmd/
    ├── subject_manifest.csv
    ├── GFBMD_skeleton_cache.h5
    └── folds_5_seed42.json
```

## ASDPose

```bash
# 1) Build the H5 from the raw pickle
python preprocessing/asdpose/build_h5.py \
    --pkl-path /path/to/"Annotated Dataset for Training.pkl" \
    --out-h5 data/asdpose/ASDpose.h5

# 2) Train (official train/test split)
python train_asdpose.py --config configs/asdpose_skeleton.yaml

# 3) Evaluate from the best checkpoint
python train_asdpose.py --config configs/asdpose_skeleton.yaml \
    --phase test --weights work_dir/asdpose_skeleton/best.pt
python evaluate.py --dataset asdpose \
    --work-dir work_dir/asdpose_skeleton \
    --out-dir work_dir/asdpose_skeleton/evaluation
```

## GFBMD

Replace `<gfbmd_root>` and `<sapiens_ckpt>` with your local paths.

```bash
# 1) Parsing maps (YOLOv8 + Sapiens; downloads yolov8n.pt on first run)
python preprocessing/gfbmd/01_extract_parsing.py \
    --dataset-root <gfbmd_root> --output-root data/gfbmd_parsing \
    --sapiens-ckpt <sapiens_ckpt> --device cuda

# 2) 8.xlsx -> .skeleton files
python preprocessing/gfbmd/02_xlsx_to_skeleton.py \
    --dataset-root <gfbmd_root> --out-root data/gfbmd_ntu

# 3) Denoise skeletons
python preprocessing/gfbmd/03_denoise_skeletons.py \
    --skeleton-dir data/gfbmd_ntu/nturgbd_raw/nturgb+d_skeletons120 \
    --names-txt data/gfbmd_ntu/statistics/skes_available_name.txt \
    --out-pkl data/gfbmd_ntu/raw_denoised_joints.pkl

# 4) Subject manifest
python preprocessing/gfbmd/04_build_manifest.py \
    --parse-root data/gfbmd_parsing --gfbmd-root <gfbmd_root> \
    --skeleton-dir data/gfbmd_ntu/nturgbd_raw/nturgb+d_skeletons120 \
    --statistics-dir data/gfbmd_ntu/statistics \
    --out-dir data/gfbmd

# 5) Skeleton cache
python preprocessing/gfbmd/05_build_cache.py \
    --manifest-csv data/gfbmd/subject_manifest.csv \
    --skes-available-name-txt data/gfbmd_ntu/statistics/skes_available_name.txt \
    --raw-denoised-joints-pkl data/gfbmd_ntu/raw_denoised_joints.pkl \
    --out-h5 data/gfbmd/GFBMD_skeleton_cache.h5 \
    --out-summary-json data/gfbmd/cache_summary.json

# 6) Fixed 5-fold splits
python preprocessing/gfbmd/06_build_folds.py \
    --manifest-csv data/gfbmd/subject_manifest.csv \
    --out-json data/gfbmd/folds_5_seed42.json

# 7) Train all 5 folds
python train_fusion.py --config configs/gfbmd_fusion.yaml

# 8) Evaluate from the best checkpoints (per fold + CV summary)
python train_fusion.py --config configs/gfbmd_fusion.yaml --phase test
python evaluate.py --dataset gfbmd \
    --work-dir work_dir/gfbmd_fusion \
    --out-dir work_dir/gfbmd_fusion/evaluation --num-folds 5
```

A single fold can be evaluated directly from a checkpoint:

```bash
python train_fusion.py --config configs/gfbmd_fusion.yaml \
    --phase test --fold 1 --weights work_dir/gfbmd_fusion/fold1/best.pt
```


## Notebooks

Step-by-step walkthroughs of the commands above:

- `notebooks/01_asdpose_training_and_evaluation.ipynb`
- `notebooks/02_gfbmd_fusion_training_and_evaluation.ipynb`

## Pretrained checkpoints

Pretrained weights (ASDPose `best.pt` and GFBMD `fold1`–`fold5` `best.pt`) are on the
[Releases](https://github.com/wondimagegn-b/DeCAPS-Net/releases) page. Use them with the
`--phase test` commands above.

## Links

- GFBMD paper: https://iopscience.iop.org/article/10.1088/1742-6596/1818/1/012201/meta
- ASDPose paper: https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2823635
- Sapiens: https://github.com/facebookresearch/sapiens
