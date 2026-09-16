"""Convert GFBMD 8.xlsx files to NTU-style .skeleton files and statistics."""

import argparse
import os
import os.path as osp
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

XLSX_NAME = "8.xlsx"
SETUP_ID = 1
CAMERA_ID = 1
REPLICATION_ID = 1

NTU_JOINTS = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight",
]

JOINT_ALIASES = {
    "midspain": "SpineMid",
    "midspine": "SpineMid",
    "spinemid": "SpineMid",
    "spinebase": "SpineBase",
    "spineshoulder": "SpineShoulder",
}

TIME_COL_PATTERNS = [
    re.compile(r"h\s*:\s*m\s*:\s*s\s*:\s*ms", re.IGNORECASE),
    re.compile(r"time", re.IGNORECASE),
]


def norm_key(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    return re.sub(r"[\s\(\)\[\]\{\}\-_/\\:;,.]+", "", s)


def is_time_col(col: str) -> bool:
    s = str(col)
    if any(pat.search(s) for pat in TIME_COL_PATTERNS):
        return True
    return norm_key(s) in ("hmssms", "hmssms)")


def safe_sort_key(name: str):
    name = str(name)
    if name.isdigit():
        return (0, int(name), name)
    return (1, 10**9, name)


def find_all_sequences(dataset_root: str) -> List[Tuple[str, str, int]]:
    seqs = []
    autism_dir = osp.join(dataset_root, "Autism", "children with ASD")
    if osp.isdir(autism_dir):
        for child in sorted(os.listdir(autism_dir), key=safe_sort_key):
            if not str(child).isdigit():
                continue
            xlsx = osp.join(autism_dir, str(child), "video", XLSX_NAME)
            if osp.isfile(xlsx):
                seqs.append((xlsx, "Autism", int(child)))

    typical_dir = osp.join(dataset_root, "Typical")
    if osp.isdir(typical_dir):
        for child in sorted(os.listdir(typical_dir), key=safe_sort_key):
            if not str(child).isdigit():
                continue
            xlsx = osp.join(typical_dir, str(child), "video", XLSX_NAME)
            if osp.isfile(xlsx):
                seqs.append((xlsx, "Typical", int(child)))
    return seqs


def read_xlsx_raw(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    return df.dropna(how="all")


def build_joint_column_map(df: pd.DataFrame) -> Dict[str, Tuple[str, str, str]]:
    cols = list(df.columns)
    joint_map = {}
    col_trip = [(i, cols[i], norm_key(cols[i])) for i in range(len(cols))]

    def resolve_joint(header_norm: str) -> str:
        if header_norm in JOINT_ALIASES:
            return JOINT_ALIASES[header_norm]
        for joint in NTU_JOINTS:
            if header_norm == norm_key(joint):
                return joint
        return ""

    i = 0
    while i < len(col_trip):
        _, col_name, col_norm = col_trip[i]
        if is_time_col(col_name):
            i += 1
            continue
        joint = resolve_joint(col_norm)
        if joint:
            if i + 2 >= len(col_trip):
                break
            joint_map[joint] = (cols[i], cols[i + 1], cols[i + 2])
            i += 3
            continue
        i += 1
    return joint_map


def dataframe_to_ntu_array(df: pd.DataFrame, joint_cols: Dict[str, Tuple[str, str, str]]) -> np.ndarray:
    arr = np.zeros((len(df), 25, 3), dtype=np.float32)
    for j_idx, joint in enumerate(NTU_JOINTS):
        if joint not in joint_cols:
            continue
        cx, cy, cz = joint_cols[joint]
        arr[:, j_idx, 0] = pd.to_numeric(df[cx], errors="coerce").fillna(0).astype(np.float32).to_numpy()
        arr[:, j_idx, 1] = pd.to_numeric(df[cy], errors="coerce").fillna(0).astype(np.float32).to_numpy()
        arr[:, j_idx, 2] = pd.to_numeric(df[cz], errors="coerce").fillna(0).astype(np.float32).to_numpy()
    return arr


def write_skeleton_file(out_path: str, joints_xyz: np.ndarray) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{joints_xyz.shape[0]}\n")
        for t in range(joints_xyz.shape[0]):
            f.write("1\n")
            f.write("1 0 0 0 0 0 0 0 0 0\n")
            f.write("25\n")
            for j in range(25):
                x, y, z = joints_xyz[t, j].tolist()
                f.write(
                    f"{x:.6f} {y:.6f} {z:.6f} "
                    f"0.000000 0.000000 0.000000 0.000000 "
                    f"1.000000 0.000000 0.000000 0.000000 2\n"
                )


def make_skeleton_name(performer_id: int, label_id: int, replication_id: int = REPLICATION_ID) -> str:
    return f"S001C001P{performer_id:03d}R{replication_id:03d}A{label_id:03d}"


def write_txt(path: str, values: List) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def convert_dataset(dataset_root: str, out_root: str) -> None:
    skes_out_dir = osp.join(out_root, "nturgbd_raw", "nturgb+d_skeletons120")
    stat_dir = osp.join(out_root, "statistics")
    os.makedirs(skes_out_dir, exist_ok=True)
    os.makedirs(stat_dir, exist_ok=True)

    seqs = find_all_sequences(dataset_root)
    print(f"Found {len(seqs)} sequences.")
    if not seqs:
        return

    performer_map = {}
    next_pid = 1
    for _, cls_name, child_id in seqs:
        key = (cls_name, child_id)
        if key not in performer_map:
            performer_map[key] = next_pid
            next_pid += 1

    ok_names, ok_setup, ok_camera, ok_performer, ok_replication, ok_label = [], [], [], [], [], []
    n_ok = 0
    n_fail = 0

    for xlsx_path, cls_name, child_id in seqs:
        try:
            df = read_xlsx_raw(xlsx_path)
            joint_cols = build_joint_column_map(df)
            if len(joint_cols) < 10:
                preview = ", ".join(str(c) for c in list(df.columns)[:40])
                raise RuntimeError(f"could not detect joints properly (found {len(joint_cols)}): {preview}")

            joints_xyz = dataframe_to_ntu_array(df, joint_cols)
            if joints_xyz.shape[0] <= 0:
                raise RuntimeError("no frames after reading")

            label_id = 1 if cls_name.lower() == "autism" else 2
            performer_id = performer_map[(cls_name, child_id)]
            skeleton_name = make_skeleton_name(performer_id, label_id)
            write_skeleton_file(osp.join(skes_out_dir, skeleton_name + ".skeleton"), joints_xyz)

            ok_names.append(skeleton_name)
            ok_setup.append(SETUP_ID)
            ok_camera.append(CAMERA_ID)
            ok_performer.append(performer_id)
            ok_replication.append(REPLICATION_ID)
            ok_label.append(label_id)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"[FAIL] {xlsx_path}: {exc}")

    print(f"Converted: {n_ok}; failed: {n_fail}")
    if n_ok == 0:
        raise RuntimeError("No sequences converted successfully.")

    write_txt(osp.join(stat_dir, "skes_available_name.txt"), ok_names)
    write_txt(osp.join(stat_dir, "setup.txt"), ok_setup)
    write_txt(osp.join(stat_dir, "camera.txt"), ok_camera)
    write_txt(osp.join(stat_dir, "performer.txt"), ok_performer)
    write_txt(osp.join(stat_dir, "replication.txt"), ok_replication)
    write_txt(osp.join(stat_dir, "label.txt"), ok_label)
    print(f"Skeletons: {skes_out_dir}")
    print(f"Statistics: {stat_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="GFBMD dataset root")
    parser.add_argument("--out-root", required=True, help="Output root for NTU-style files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(args.dataset_root, args.out_root)
