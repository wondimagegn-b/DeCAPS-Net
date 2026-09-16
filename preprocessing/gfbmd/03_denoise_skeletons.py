"""Build raw_denoised_joints.pkl from GFBMD NTU-style .skeleton files."""

import argparse
import os
import pickle
from typing import List

import numpy as np


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_skeleton_joints(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        num_frames = int(f.readline().strip())
        joints = np.zeros((num_frames, 75), dtype=np.float32)

        for t in range(num_frames):
            num_bodies = int(f.readline().strip())
            if num_bodies < 1:
                raise ValueError(f"{path}: frame {t} has no bodies")

            for body_idx in range(num_bodies):
                f.readline()
                num_joints = int(f.readline().strip())
                body_joints = np.zeros((num_joints, 3), dtype=np.float32)
                for j in range(num_joints):
                    values = f.readline().split()
                    if len(values) < 3:
                        raise ValueError(f"{path}: invalid joint row at frame {t}, joint {j}")
                    body_joints[j] = np.asarray(values[:3], dtype=np.float32)

                if body_idx == 0:
                    if num_joints != 25:
                        raise ValueError(f"{path}: expected 25 joints, got {num_joints}")
                    joints[t] = body_joints.reshape(-1)
    return joints


def build_denoised_joints(skeleton_dir: str, names_txt: str, out_pkl: str) -> None:
    names = read_lines(names_txt)
    sequences = []
    for name in names:
        path = os.path.join(skeleton_dir, name + ".skeleton")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing skeleton file: {path}")
        sequences.append(read_skeleton_joints(path))

    out_dir = os.path.dirname(out_pkl)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(sequences, f, pickle.HIGHEST_PROTOCOL)

    total_frames = int(sum(seq.shape[0] for seq in sequences))
    print(f"Saved {len(sequences)} sequences ({total_frames} frames): {out_pkl}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton-dir", required=True, help="Directory containing .skeleton files")
    parser.add_argument("--names-txt", required=True, help="skes_available_name.txt")
    parser.add_argument("--out-pkl", required=True, help="Output raw_denoised_joints.pkl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_denoised_joints(args.skeleton_dir, args.names_txt, args.out_pkl)
