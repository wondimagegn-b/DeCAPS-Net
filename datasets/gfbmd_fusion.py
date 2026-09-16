"""GFBMD subject-level dataset: colorized parsing maps + skeleton cache.

Works with the preprocessing outputs:
- subject manifest CSV (one row per subject, incl. parse_color_dir)
- folds JSON (subject-level CV splits)
- skeleton cache H5 (x_all [N, T, 75], y, seq_len, subject_uid)

Each item returns:
    parse_x: (T_parse, 3, H, W) float32 tensor
    skel_x:  (3, window_size, 25, 1) float32 tensor
    label:   int (ASD=1, TD=0)
    index:   int
    meta:    dict (subject_uid, label_name, child_id, ...)
"""

import os
import csv
import json
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from datasets.transforms import valid_crop_resize, random_rot

try:
    import h5py
except ImportError:
    h5py = None


def _to_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _read_manifest_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _decode_if_bytes(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def _sample_evenly(paths, n_frames):
    if len(paths) == 0:
        return []
    if n_frames <= 1:
        return [paths[0]]
    idxs = np.linspace(0, len(paths) - 1, n_frames)
    idxs = np.round(idxs).astype(int)
    idxs = np.clip(idxs, 0, len(paths) - 1)
    return [paths[i] for i in idxs]


class GFBMDFusion(Dataset):
    """Parsing-map + skeleton subject-level dataset for GFBMD."""

    def __init__(
        self,
        manifest_csv,
        folds_json,
        skeleton_h5,
        split="train",              # "train" or "val" ("test" aliases to "val")
        fold=1,                     # 1..5
        # parsing branch
        parse_n_frames=8,
        parse_img_size=224,         # int or (H, W)
        parse_imagenet_norm=True,
        # skeleton branch
        p_interval=(1.0,),
        window_size=64,
        random_rot=False,
        # misc
        debug=False,
    ):
        super().__init__()

        if h5py is None:
            raise ImportError("h5py is required. Install with: pip install h5py")

        self.manifest_csv = manifest_csv
        self.folds_json = folds_json
        self.skeleton_h5 = skeleton_h5
        self.split = "val" if split == "test" else split
        self.fold = int(fold)

        self.parse_n_frames = int(parse_n_frames)
        if isinstance(parse_img_size, int):
            self.parse_h = int(parse_img_size)
            self.parse_w = int(parse_img_size)
        else:
            self.parse_h = int(parse_img_size[0])
            self.parse_w = int(parse_img_size[1])
        self.parse_imagenet_norm = bool(parse_imagenet_norm)

        self.p_interval = p_interval
        self.window_size = int(window_size)
        self.random_rot = bool(random_rot)
        self.debug = bool(debug)

        # ImageNet normalization (for the pretrained parsing-map encoder)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        # HDF5 handle (lazy-open for dataloader workers)
        self._h5 = None
        self._x_all = None
        self._seq_len = None
        self._y = None

        # Build subject list (alignment between manifest + folds + skeleton cache)
        self.samples = self._build_samples()

        if self.debug:
            self.samples = self.samples[: min(20, len(self.samples))]

        self.label = np.array([s["label"] for s in self.samples], dtype=np.int64)
        self.sample_name = [s["subject_uid"] for s in self.samples]

        print(
            f"[GFBMDFusion] split={self.split} fold={self.fold} "
            f"subjects={len(self.samples)} ASD={(self.label == 1).sum()} TD={(self.label == 0).sum()}"
        )

    # -------------------------------------------------
    # Setup / metadata loading
    # -------------------------------------------------
    def _build_samples(self):
        if not os.path.isfile(self.manifest_csv):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")
        if not os.path.isfile(self.folds_json):
            raise FileNotFoundError(f"Folds JSON not found: {self.folds_json}")
        if not os.path.isfile(self.skeleton_h5):
            raise FileNotFoundError(f"Skeleton cache H5 not found: {self.skeleton_h5}")

        manifest_rows = _read_manifest_csv(self.manifest_csv)
        manifest_by_uid = {}
        for r in manifest_rows:
            uid = str(r.get("subject_uid", "")).strip()
            if not uid:
                continue
            manifest_by_uid[uid] = r

        folds_data = _read_json(self.folds_json)
        folds = folds_data.get("folds", [])
        if not folds:
            raise RuntimeError("folds_json has no 'folds'.")

        fold_obj = None
        for fd in folds:
            if int(fd.get("fold", -1)) == self.fold:
                fold_obj = fd
                break
        if fold_obj is None:
            raise RuntimeError(f"Fold {self.fold} not found in folds_json.")

        if self.split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        uids = fold_obj["train_subject_uids"] if self.split == "train" else fold_obj["val_subject_uids"]

        # Read skeleton-cache metadata once (no long-lived h5 handle yet)
        with h5py.File(self.skeleton_h5, "r") as h5:
            if "subject_uid" not in h5 or "x_all" not in h5 or "y" not in h5:
                raise RuntimeError("Skeleton cache missing required datasets: subject_uid / x_all / y")

            cache_uids = [_decode_if_bytes(x) for x in h5["subject_uid"][:]]
            cache_y = np.array(h5["y"][:], dtype=np.int64)

            x_shape = h5["x_all"].shape  # [N, T, D]
            if len(x_shape) != 3:
                raise RuntimeError(f"x_all must be [N, T, D], got {x_shape}")
            self._cache_Tmax = int(x_shape[1])
            self._cache_D = int(x_shape[2])  # 75 or 150 expected

        uid_to_cache_idx = {uid: i for i, uid in enumerate(cache_uids)}

        samples = []
        missing = []
        label_mismatch = []
        parse_missing = []

        for uid in uids:
            if uid not in manifest_by_uid:
                missing.append(f"{uid} missing in manifest")
                continue
            if uid not in uid_to_cache_idx:
                missing.append(f"{uid} missing in skeleton cache")
                continue

            r = manifest_by_uid[uid]
            cache_idx = uid_to_cache_idx[uid]

            # labels
            man_label = _to_int(r.get("label", None), None)
            if man_label not in (0, 1):
                lname = str(r.get("label_name", "")).strip().upper()
                man_label = 1 if lname == "ASD" else 0 if lname == "TD" else None
            if man_label not in (0, 1):
                raise RuntimeError(f"Cannot parse label for {uid} from manifest.")

            cache_label = int(cache_y[cache_idx])
            if cache_label != man_label:
                label_mismatch.append((uid, man_label, cache_label))
                continue

            # parsing maps
            parse_dir = str(r.get("parse_color_dir", "")).strip()
            if not parse_dir or (not os.path.isdir(parse_dir)):
                parse_missing.append((uid, parse_dir))
                continue

            frame_paths = sorted(glob.glob(os.path.join(parse_dir, "*.png")))
            if len(frame_paths) == 0:
                parse_missing.append((uid, parse_dir))
                continue

            samples.append({
                "subject_uid": uid,
                "label": int(man_label),
                "label_name": str(r.get("label_name", "ASD" if man_label == 1 else "TD")),
                "child_id": str(r.get("child_id", "")),
                "parse_color_dir": parse_dir,
                "frame_paths": frame_paths,
                "cache_idx": int(cache_idx),
                "skeleton_name": str(r.get("skeleton_name", "")),
                "performer_id": _to_int(r.get("performer_id", 0), 0),
            })

        if missing:
            raise RuntimeError(f"Missing alignment entries. Example: {missing[:5]}")
        if label_mismatch:
            raise RuntimeError(f"Manifest/cache label mismatch. Example: {label_mismatch[:5]}")
        if parse_missing:
            raise RuntimeError(f"Missing parse PNGs. Example: {parse_missing[:5]}")

        return samples

    # -------------------------------------------------
    # HDF5 lazy open
    # -------------------------------------------------
    def _open_h5_if_needed(self):
        if self._h5 is not None:
            return
        self._h5 = h5py.File(self.skeleton_h5, "r")
        self._x_all = self._h5["x_all"]
        self._y = self._h5["y"]
        self._seq_len = self._h5["seq_len"] if "seq_len" in self._h5 else None

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

    # -------------------------------------------------
    # Parsing branch
    # -------------------------------------------------
    def _read_parse_image(self, path):
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read parse image: {path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.parse_w, self.parse_h), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        if self.parse_imagenet_norm:
            img = (img - self.mean) / self.std

        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return img  # [3, H, W]

    def _load_parse_tensor(self, frame_paths):
        paths = _sample_evenly(frame_paths, self.parse_n_frames)
        frames = [self._read_parse_image(p) for p in paths]
        x = np.stack(frames, axis=0).astype(np.float32)  # [T, 3, H, W]
        return torch.from_numpy(x)

    # -------------------------------------------------
    # Skeleton branch
    # -------------------------------------------------
    def _load_skeleton_tensor(self, cache_idx):
        self._open_h5_if_needed()

        # x_all row is padded [Tmax, D]
        x = np.array(self._x_all[cache_idx], dtype=np.float32)  # [Tmax, 75] or [Tmax, 150]

        if self._seq_len is not None:
            valid_T = int(self._seq_len[cache_idx])
        else:
            valid_T = int(np.sum(np.abs(x).sum(axis=1) > 0))

        valid_T = max(1, min(valid_T, x.shape[0]))
        x = x[:valid_T]  # [T, D]

        D = x.shape[1]
        if D == 75:
            # [T, 25, 3] -> [3, T, 25, 1]
            x = x.reshape(valid_T, 25, 3).transpose(2, 0, 1)[:, :, :, None]
        elif D == 150:
            # [T, 2, 25, 3] -> [3, T, 25, 2]
            x = x.reshape(valid_T, 2, 25, 3).transpose(3, 0, 2, 1)
        else:
            raise RuntimeError(f"Unexpected skeleton feature dim {D} (expected 75 or 150)")

        # Temporal center-crop to valid frames + resize to window_size
        data_numpy = valid_crop_resize(
            x, valid_frame_num=valid_T, p_interval=self.p_interval, window=self.window_size
        )

        if self.random_rot:
            data_numpy = random_rot(data_numpy)

        if torch.is_tensor(data_numpy):
            return data_numpy.float()  # [C, T, V, M]
        return torch.from_numpy(np.asarray(data_numpy, dtype=np.float32))  # [C, T, V, M]

    # -------------------------------------------------
    # Dataset API
    # -------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]

        parse_x = self._load_parse_tensor(s["frame_paths"])           # [T_parse, 3, H, W]
        skel_x = self._load_skeleton_tensor(s["cache_idx"])           # [C, T, V, M]
        label = int(s["label"])

        meta = {
            "subject_uid": s["subject_uid"],
            "label_name": s["label_name"],
            "child_id": s["child_id"],
            "skeleton_name": s["skeleton_name"],
            "performer_id": int(s["performer_id"]),
            "parse_num_available_frames": int(len(s["frame_paths"])),
            "cache_idx": int(s["cache_idx"]),
        }

        return parse_x, skel_x, label, index, meta
