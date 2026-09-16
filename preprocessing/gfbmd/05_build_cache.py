"""Build the split-free GFBMD skeleton cache aligned to subject_manifest.csv."""

import argparse
import csv
import json
import os
import pickle

import h5py
import numpy as np


def read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_manifest_csv(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def remove_nan_frames(seq: np.ndarray) -> np.ndarray:
    if seq.size == 0:
        return seq
    return seq[~np.isnan(seq).any(axis=1)]


def ensure_single_body_75(seq: np.ndarray) -> np.ndarray:
    if seq.ndim != 2:
        raise ValueError(f"Expected 2D sequence [T,D], got shape={seq.shape}")
    if seq.shape[1] == 75:
        return seq.astype(np.float32, copy=False)
    if seq.shape[1] == 150:
        return seq[:, :75].astype(np.float32, copy=False)
    raise ValueError(f"Unexpected skeleton feature dim: {seq.shape[1]} (expected 75 or 150)")


def center_sequence(seq75: np.ndarray, center_joint_idx: int = 0) -> np.ndarray:
    if seq75.size == 0:
        return seq75
    start = center_joint_idx * 3
    end = start + 3
    origin = None
    for t in range(seq75.shape[0]):
        center = seq75[t, start:end]
        if np.any(center != 0):
            origin = center.copy()
            break
    if origin is None:
        return seq75
    return seq75 - np.tile(origin, 25).astype(np.float32)


def pad_or_truncate(seq: np.ndarray, max_frames: int):
    T = seq.shape[0]
    if T >= max_frames:
        return seq[:max_frames].astype(np.float32), int(max_frames)
    out = np.zeros((max_frames, seq.shape[1]), dtype=np.float32)
    out[:T] = seq
    return out, int(T)


def one_hot_2class(y01: np.ndarray) -> np.ndarray:
    y = np.zeros((len(y01), 2), dtype=np.float32)
    y[np.arange(len(y01)), y01] = 1.0
    return y


def build_cache(args) -> None:
    manifest = read_manifest_csv(args.manifest_csv)
    if not manifest:
        raise RuntimeError("Manifest is empty.")

    manifest_ready = [r for r in manifest if to_int(r.get("fusion_ready", 0)) == 1]
    if len(manifest_ready) != len(manifest):
        print(f"[WARN] Using fusion_ready=1 rows: {len(manifest_ready)}/{len(manifest)}")
    manifest_ready = sorted(manifest_ready, key=lambda r: r["subject_uid"])

    subject_uids = [r["subject_uid"] for r in manifest_ready]
    if len(subject_uids) != len(set(subject_uids)):
        raise RuntimeError("Duplicate subject_uid found in manifest.")
    skeleton_names = [r["skeleton_name"] for r in manifest_ready]
    if len(skeleton_names) != len(set(skeleton_names)):
        raise RuntimeError("Duplicate skeleton_name found in manifest.")

    skes_names = read_lines(args.skes_available_name_txt)
    with open(args.raw_denoised_joints_pkl, "rb") as f:
        raw_denoised_joints = pickle.load(f)
    if len(skes_names) != len(raw_denoised_joints):
        raise RuntimeError(
            f"Mismatch: {len(skes_names)} skeleton names but {len(raw_denoised_joints)} sequences."
        )
    name_to_idx = {name: i for i, name in enumerate(skes_names)}
    if len(name_to_idx) != len(skes_names):
        raise RuntimeError("Duplicate skeleton names found in skes_available_name.txt")

    seqs = []
    seq_lens_before = []
    seq_lens_after = []
    y = []
    subject_uid_arr = []
    label_name_arr = []
    child_id_arr = []
    parse_color_dir_arr = []
    parse_frame_count_arr = []
    skeleton_name_arr = []
    skeleton_path_arr = []
    performer_id_arr = []
    action_code_arr = []
    missing_names = []
    label_mismatch = []

    for row in manifest_ready:
        subject_uid = row["subject_uid"]
        label = to_int(row["label"], -1)
        label_name = row["label_name"]
        skeleton_name = row["skeleton_name"]
        if skeleton_name not in name_to_idx:
            missing_names.append((subject_uid, skeleton_name))
            continue

        action_code = to_int(row.get("action_code_1based", 0), 0)
        if label_name == "ASD" and action_code != 1:
            label_mismatch.append((subject_uid, label_name, action_code))
        if label_name == "TD" and action_code != 2:
            label_mismatch.append((subject_uid, label_name, action_code))

        seq = np.asarray(raw_denoised_joints[name_to_idx[skeleton_name]], dtype=np.float32)
        if seq.ndim != 2:
            raise RuntimeError(f"{skeleton_name}: expected 2D sequence, got {seq.shape}")
        seq_lens_before.append(int(seq.shape[0]))
        seq = remove_nan_frames(seq)
        seq = ensure_single_body_75(seq)
        seq = center_sequence(seq, center_joint_idx=args.center_joint_index)
        if seq.shape[0] == 0:
            raise RuntimeError(f"{skeleton_name}: sequence became empty after cleaning.")
        seq_lens_after.append(int(seq.shape[0]))
        seqs.append(seq)

        y.append(label)
        subject_uid_arr.append(subject_uid)
        label_name_arr.append(label_name)
        child_id_arr.append(str(row.get("child_id", "")))
        parse_color_dir_arr.append(row.get("parse_color_dir", ""))
        parse_frame_count_arr.append(to_int(row.get("parse_frame_count", 0), 0))
        skeleton_name_arr.append(skeleton_name)
        skeleton_path_arr.append(row.get("skeleton_path", ""))
        performer_id_arr.append(to_int(row.get("performer_id", 0), 0))
        action_code_arr.append(action_code)

    if missing_names:
        raise RuntimeError(f"Manifest skeleton names missing from skes_available_name.txt: {missing_names[:3]}")
    if label_mismatch:
        raise RuntimeError(f"Label mismatch found: {label_mismatch[:3]}")
    if not seqs:
        raise RuntimeError("No sequences collected for cache.")

    observed_max_frames = max(seq.shape[0] for seq in seqs)
    max_frames = observed_max_frames if args.cache_max_frames is None else int(args.cache_max_frames)
    x_all = np.zeros((len(seqs), max_frames, 75), dtype=np.float32)
    seq_len = np.zeros((len(seqs),), dtype=np.int32)
    for i, seq in enumerate(seqs):
        clip, length = pad_or_truncate(seq, max_frames)
        x_all[i] = clip
        seq_len[i] = length

    y = np.asarray(y, dtype=np.int64)
    y_onehot = one_hot_2class(y)

    out_h5_dir = os.path.dirname(args.out_h5)
    if out_h5_dir:
        os.makedirs(out_h5_dir, exist_ok=True)
    out_summary_dir = os.path.dirname(args.out_summary_json)
    if out_summary_dir:
        os.makedirs(out_summary_dir, exist_ok=True)
    if os.path.exists(args.out_h5):
        os.remove(args.out_h5)

    dt_str = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.out_h5, "w") as h5:
        h5.create_dataset("x_all", data=x_all, compression="lzf", shuffle=True)
        h5.create_dataset("seq_len", data=seq_len)
        h5.create_dataset("y", data=y.astype(np.int8))
        h5.create_dataset("y_onehot", data=y_onehot)
        h5.create_dataset("subject_uid", data=np.array(subject_uid_arr, dtype=object), dtype=dt_str)
        h5.create_dataset("label_name", data=np.array(label_name_arr, dtype=object), dtype=dt_str)
        h5.create_dataset("child_id", data=np.array(child_id_arr, dtype=object), dtype=dt_str)
        h5.create_dataset("parse_color_dir", data=np.array(parse_color_dir_arr, dtype=object), dtype=dt_str)
        h5.create_dataset("parse_frame_count", data=np.asarray(parse_frame_count_arr, dtype=np.int32))
        h5.create_dataset("skeleton_name", data=np.array(skeleton_name_arr, dtype=object), dtype=dt_str)
        h5.create_dataset("skeleton_path", data=np.array(skeleton_path_arr, dtype=object), dtype=dt_str)
        h5.create_dataset("performer_id", data=np.asarray(performer_id_arr, dtype=np.int32))
        h5.create_dataset("action_code_1based", data=np.asarray(action_code_arr, dtype=np.int16))
        h5.attrs["num_subjects"] = len(seqs)
        h5.attrs["feature_dim"] = 75
        h5.attrs["cache_max_frames"] = max_frames
        h5.attrs["observed_max_frames"] = observed_max_frames
        h5.attrs["label_convention"] = "ASD=1, TD=0"
        h5.attrs["center_joint_index"] = int(args.center_joint_index)
        h5.attrs["center_joint_name"] = "SpineBase" if args.center_joint_index == 0 else f"joint_{args.center_joint_index}"
        h5.attrs["sequence_layout"] = "T x 75 (1 body x 25 joints x 3 coords), padded with zeros"

    summary = {
        "input": {
            "manifest_csv": args.manifest_csv,
            "skes_available_name_txt": args.skes_available_name_txt,
            "raw_denoised_joints_pkl": args.raw_denoised_joints_pkl,
        },
        "output": {"skeleton_cache_h5": args.out_h5},
        "counts": {
            "num_subjects": int(len(seqs)),
            "asd_subjects": int((y == 1).sum()),
            "td_subjects": int((y == 0).sum()),
        },
        "sequence_lengths": {
            "before_clean_min": int(np.min(seq_lens_before)),
            "before_clean_max": int(np.max(seq_lens_before)),
            "before_clean_mean": float(np.mean(seq_lens_before)),
            "after_clean_min": int(np.min(seq_lens_after)),
            "after_clean_max": int(np.max(seq_lens_after)),
            "after_clean_mean": float(np.mean(seq_lens_after)),
            "cache_max_frames": int(max_frames),
        },
        "config": {
            "CENTER_JOINT_INDEX": int(args.center_joint_index),
            "CACHE_MAX_FRAMES": None if args.cache_max_frames is None else int(args.cache_max_frames),
            "label_convention": "ASD=1, TD=0",
        },
    }
    with open(args.out_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Cache: {args.out_h5}")
    print(f"Summary: {args.out_summary_json}")
    print(json.dumps(summary["counts"], indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--skes-available-name-txt", required=True)
    parser.add_argument("--raw-denoised-joints-pkl", required=True)
    parser.add_argument("--out-h5", required=True)
    parser.add_argument("--out-summary-json", required=True)
    parser.add_argument("--cache-max-frames", type=int, default=None)
    parser.add_argument("--center-joint-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
