"""Build the ASDPose raw-clip HDF5 file from the annotated PKL."""

import argparse
import os
import pickle
from collections import Counter, defaultdict
from typing import Tuple

import h5py
import numpy as np


def infer_wh(img_shape, kp_xy) -> Tuple[float, float]:
    a = float(img_shape[0])
    b = float(img_shape[1])
    x_max = float(np.nanmax(kp_xy[..., 0])) if kp_xy.size else 0.0
    y_max = float(np.nanmax(kp_xy[..., 1])) if kp_xy.size else 0.0
    ok1 = (x_max <= a * 1.2) and (y_max <= b * 1.2)
    ok2 = (x_max <= b * 1.2) and (y_max <= a * 1.2)
    if ok1 and not ok2:
        return a, b
    if ok2 and not ok1:
        return b, a
    return a, b


def parse_identifier(identifier: str):
    parts = identifier.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected identifier format: {identifier}")
    child_id = int(parts[0])
    assessment_id = int(parts[1])
    video_id = int(parts[2])
    start_frame = int(parts[-2])
    end_frame = int(parts[-1])
    action_from_identifier = "_".join(parts[3:-2])
    return child_id, assessment_id, video_id, action_from_identifier, start_frame, end_frame


def normalize_clip(kp_xy: np.ndarray, kp_score: np.ndarray, img_shape) -> Tuple[np.ndarray, float, float]:
    kp = np.asarray(kp_xy, dtype=np.float32).copy()
    ks = np.clip(np.asarray(kp_score, dtype=np.float32).copy(), 0.0, 1.0)
    missing = ks <= 0.0
    kp[missing, 0] = 0.0
    kp[missing, 1] = 0.0

    W, H = infer_wh(img_shape, kp)
    W = max(float(W), 1.0)
    H = max(float(H), 1.0)
    kp[..., 0] = np.clip(kp[..., 0] / W, 0.0, 1.0)
    kp[..., 1] = np.clip(kp[..., 1] / H, 0.0, 1.0)
    feat = np.concatenate([kp, ks[..., None]], axis=-1).astype(np.float32)
    return feat, W, H


def build_h5(pkl_path: str, out_h5: str, compression: str = "lzf", gzip_level: int = 4, keep_duplicates: bool = True) -> None:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Missing PKL: {pkl_path}")
    out_dir = os.path.dirname(out_h5)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    anns = data["annotations"]
    train_ids = set(data["split"]["train"])
    test_ids = set(data["split"]["test"])

    ids = [a["identifier"] for a in anns]
    counts = Counter(ids)
    duplicated_ids = [k for k, v in counts.items() if v > 1]
    print(f"Annotations: {len(anns)}; unique identifiers: {len(counts)}; duplicated identifiers: {len(duplicated_ids)}")

    selected = []
    seen = set()
    for src_idx, ann in enumerate(anns):
        identifier = ann["identifier"]
        if (not keep_duplicates) and identifier in seen:
            continue
        seen.add(identifier)
        if identifier in train_ids:
            split = 0
        elif identifier in test_ids:
            split = 1
        else:
            continue
        selected.append((src_idx, split))

    N = len(selected)
    if N == 0:
        raise RuntimeError("No annotations matched the official train/test split.")
    print(f"Selected rows: {N}")

    lengths = np.zeros(N, dtype=np.int32)
    for i, (src_idx, _) in enumerate(selected):
        lengths[i] = int(np.asarray(anns[src_idx]["keypoint"]).shape[0])
    seq_indptr = np.zeros(N + 1, dtype=np.int64)
    seq_indptr[1:] = np.cumsum(lengths, dtype=np.int64)
    total_frames = int(seq_indptr[-1])
    print(
        f"Frames: {total_frames}; T min={lengths.min()}, median={int(np.median(lengths))}, "
        f"mean={lengths.mean():.2f}, max={lengths.max()}"
    )

    if os.path.exists(out_h5):
        os.remove(out_h5)
    dt_str = h5py.string_dtype(encoding="utf-8")
    h5_kwargs = {}
    if compression == "lzf":
        h5_kwargs["compression"] = "lzf"
    elif compression == "gzip":
        h5_kwargs["compression"] = "gzip"
        h5_kwargs["compression_opts"] = int(gzip_level)
    elif compression != "none":
        raise ValueError(f"Unknown compression: {compression}")

    with h5py.File(out_h5, "w") as h5:
        h5.attrs["dataset_name"] = "ASDPose_Annotated_Dataset_for_Training"
        h5.attrs["num_sequences"] = int(N)
        h5.attrs["total_frames"] = int(total_frames)
        h5.attrs["num_joints"] = 17
        h5.attrs["feature_channels"] = 3
        h5.attrs["keep_duplicates"] = int(keep_duplicates)
        h5.attrs["compression"] = str(compression)
        h5.attrs["has_local_frame_labels"] = 0
        h5.attrs["local_frame_labels_note"] = (
            "This PKL provides clip skeletons and global event start/end, "
            "but not per-frame global indices within each clip. "
            "Local frame labels must be reconstructed later if clip-to-video alignment is available."
        )

        chunk_frames = min(max(1024, lengths.max()), 8192)
        frames_ds = h5.create_dataset(
            "frames",
            shape=(total_frames, 17, 3),
            dtype="float32",
            chunks=(chunk_frames, 17, 3),
            shuffle=True,
            **h5_kwargs,
        )
        h5.create_dataset("seq_indptr", data=seq_indptr, dtype="int64")
        h5.create_dataset("seq_length", data=lengths, dtype="int32")

        split_ds = h5.create_dataset("split", shape=(N,), dtype="uint8")
        y_ds = h5.create_dataset("binary_label", shape=(N,), dtype="uint8")
        id_ds = h5.create_dataset("identifier", shape=(N,), dtype=dt_str)
        child_ds = h5.create_dataset("child_id", shape=(N,), dtype="int32")
        assess_ds = h5.create_dataset("assessment_id", shape=(N,), dtype="int32")
        video_ds = h5.create_dataset("video_id", shape=(N,), dtype="int32")
        action_ds = h5.create_dataset("action_name", shape=(N,), dtype=dt_str)
        action_id_ds = h5.create_dataset("action_name_from_identifier", shape=(N,), dtype=dt_str)
        start_ds = h5.create_dataset("event_start_frame_global", shape=(N,), dtype="int32")
        end_ds = h5.create_dataset("event_end_frame_global", shape=(N,), dtype="int32")
        event_len_ds = h5.create_dataset("event_length", shape=(N,), dtype="int32")
        fps_ds = h5.create_dataset("fps", shape=(N,), dtype="float32")
        img_w_ds = h5.create_dataset("img_w", shape=(N,), dtype="float32")
        img_h_ds = h5.create_dataset("img_h", shape=(N,), dtype="float32")
        img_shape_ds = h5.create_dataset("img_shape_raw", shape=(N,), dtype=dt_str)
        src_idx_ds = h5.create_dataset("source_annotation_index", shape=(N,), dtype="int32")
        dup_rank_ds = h5.create_dataset("duplicate_rank_for_identifier", shape=(N,), dtype="int32")

        train_row_idx = []
        test_row_idx = []
        dup_seen = defaultdict(int)
        for i, (src_idx, split) in enumerate(selected):
            ann = anns[src_idx]
            feat, W, H = normalize_clip(
                np.asarray(ann["keypoint"], dtype=np.float32),
                np.asarray(ann["keypoint_score"], dtype=np.float32),
                ann.get("img_shape", (1, 1)),
            )
            start = int(seq_indptr[i])
            end = int(seq_indptr[i + 1])
            if end - start != feat.shape[0]:
                raise RuntimeError(f"Frame count mismatch for annotation index {src_idx}")
            frames_ds[start:end] = feat

            identifier = str(ann["identifier"])
            child_id, assessment_id, video_id, action_from_id, start_f, end_f = parse_identifier(identifier)
            split_ds[i] = np.uint8(split)
            y_ds[i] = np.uint8(int(ann["binary_label"]))
            id_ds[i] = identifier
            child_ds[i] = child_id
            assess_ds[i] = assessment_id
            video_ds[i] = video_id
            action_ds[i] = str(ann.get("action_name", ""))
            action_id_ds[i] = str(action_from_id)
            start_ds[i] = int(start_f)
            end_ds[i] = int(end_f)
            event_len_ds[i] = int(end_f - start_f + 1)
            fps_ds[i] = float(ann.get("fps", 0.0))
            img_w_ds[i] = float(W)
            img_h_ds[i] = float(H)
            img_shape_ds[i] = str(ann.get("img_shape", ""))
            src_idx_ds[i] = int(src_idx)
            dup_rank = dup_seen[identifier]
            dup_rank_ds[i] = int(dup_rank)
            dup_seen[identifier] += 1
            if split == 0:
                train_row_idx.append(i)
            else:
                test_row_idx.append(i)
            if (i + 1) % 2000 == 0 or (i + 1) == N:
                print(f"Wrote {i + 1}/{N}")

        h5.create_dataset("train_row_indices", data=np.array(train_row_idx, dtype=np.int32))
        h5.create_dataset("test_row_indices", data=np.array(test_row_idx, dtype=np.int32))
        h5.create_dataset("split_train_ids", data=np.array(sorted(train_ids), dtype=object), dtype=dt_str)
        h5.create_dataset("split_test_ids", data=np.array(sorted(test_ids), dtype=object), dtype=dt_str)
        train_children = sorted({int(identifier.split("_", 1)[0]) for identifier in train_ids})
        test_children = sorted({int(identifier.split("_", 1)[0]) for identifier in test_ids})
        h5.create_dataset("split_train_child_ids", data=np.array(train_children, dtype=np.int32))
        h5.create_dataset("split_test_child_ids", data=np.array(test_children, dtype=np.int32))

    print(f"Saved: {out_h5}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl-path", required=True, help="Annotated Dataset for Training.pkl")
    parser.add_argument("--out-h5", required=True, help="Output ASDPose H5 file")
    parser.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument("--drop-duplicates", action="store_true", help="Keep only the first row per identifier")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_h5(
        pkl_path=args.pkl_path,
        out_h5=args.out_h5,
        compression=args.compression,
        gzip_level=args.gzip_level,
        keep_duplicates=not args.drop_duplicates,
    )
