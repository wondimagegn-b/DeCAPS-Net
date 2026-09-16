"""Build fixed subject-wise stratified CV folds for GFBMD."""

import argparse
import csv
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold


def to_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def read_manifest_csv(path: str):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_label01(row) -> int:
    if "label" in row and str(row["label"]).strip() != "":
        value = to_int(row["label"], None)
        if value in (0, 1):
            return value
    name = str(row.get("label_name", "")).strip().upper()
    if name == "ASD":
        return 1
    if name == "TD":
        return 0
    raise ValueError(f"Cannot parse label from row: {row}")


def build_folds(manifest_csv: str, out_json: str, num_folds: int, random_seed: int) -> None:
    if not os.path.isfile(manifest_csv):
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
    rows = read_manifest_csv(manifest_csv)
    if not rows:
        raise RuntimeError("Manifest CSV is empty.")

    if "fusion_ready" in rows[0]:
        rows = [r for r in rows if to_int(r.get("fusion_ready", 0), 0) == 1]
    if not rows:
        raise RuntimeError("No rows left after filtering fusion_ready=1.")
    rows = sorted(rows, key=lambda r: str(r.get("subject_uid", "")).strip())

    subject_uids = []
    labels = []
    label_names = []
    parse_frame_counts = []
    skeleton_names = []
    seen = set()
    for row in rows:
        uid = str(row.get("subject_uid", "")).strip()
        if not uid:
            raise RuntimeError(f"Missing subject_uid in row: {row}")
        if uid in seen:
            raise RuntimeError(f"Duplicate subject_uid found in manifest: {uid}")
        seen.add(uid)
        label = get_label01(row)
        subject_uids.append(uid)
        labels.append(label)
        label_names.append(str(row.get("label_name", "ASD" if label == 1 else "TD")).strip())
        parse_frame_counts.append(to_int(row.get("parse_frame_count", 0), 0))
        skeleton_names.append(str(row.get("skeleton_name", "")).strip())

    labels = np.asarray(labels, dtype=np.int64)
    n_total = len(subject_uids)
    n_asd = int((labels == 1).sum())
    n_td = int((labels == 0).sum())
    if n_asd == 0 or n_td == 0:
        raise RuntimeError(f"Need both classes for stratified CV. Got ASD={n_asd}, TD={n_td}")

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n_total), labels), start=1):
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        folds.append({
            "fold": fold_idx,
            "train_indices": train_idx,
            "val_indices": val_idx,
            "train_subject_uids": [subject_uids[i] for i in train_idx],
            "val_subject_uids": [subject_uids[i] for i in val_idx],
            "counts": {
                "train_total": int(len(train_idx)),
                "train_asd": int((y_train == 1).sum()),
                "train_td": int((y_train == 0).sum()),
                "val_total": int(len(val_idx)),
                "val_asd": int((y_val == 1).sum()),
                "val_td": int((y_val == 0).sum()),
            },
        })

    subjects = [
        {
            "index": i,
            "subject_uid": subject_uids[i],
            "label": int(labels[i]),
            "label_name": label_names[i],
            "parse_frame_count": int(parse_frame_counts[i]),
            "skeleton_name": skeleton_names[i],
        }
        for i in range(n_total)
    ]
    out = {
        "meta": {
            "num_folds": num_folds,
            "random_seed": random_seed,
            "split_type": "subject-wise stratified k-fold",
            "label_convention": "ASD=1, TD=0",
            "num_subjects": n_total,
            "num_asd": n_asd,
            "num_td": n_td,
            "manifest_csv": manifest_csv,
            "used_only_fusion_ready": True,
        },
        "subjects": subjects,
        "folds": folds,
    }

    out_dir = os.path.dirname(out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Folds: {out_json}")
    for fold in folds:
        counts = fold["counts"]
        print(
            f"Fold {fold['fold']}: train={counts['train_total']} "
            f"(ASD={counts['train_asd']}, TD={counts['train_td']}), "
            f"val={counts['val_total']} (ASD={counts['val_asd']}, TD={counts['val_td']})"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_folds(args.manifest_csv, args.out_json, args.num_folds, args.random_seed)
