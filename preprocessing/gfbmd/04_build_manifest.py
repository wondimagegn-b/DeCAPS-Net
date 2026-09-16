"""Build the GFBMD subject manifest aligning parsing maps and skeleton files."""

import argparse
import csv
import glob
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np

_LABEL_RE = re.compile(r"A(\d{3})")
_PERF_RE = re.compile(r"P(\d{3})")


def safe_numeric_sort_key(name: str):
    s = str(name)
    if s.isdigit():
        return (0, int(s), s)
    return (1, 10**9, s.lower())


def make_skeleton_name(performer_id: int, label_id_1based: int, replication_id: int = 1) -> str:
    return f"S001C001P{performer_id:03d}R{replication_id:03d}A{label_id_1based:03d}"


def load_txt_lines(path: str) -> List[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_action_code(skeleton_name: str) -> int:
    match = _LABEL_RE.search(skeleton_name)
    if not match:
        raise ValueError(f"Cannot parse A### from skeleton name: {skeleton_name}")
    return int(match.group(1))


def parse_performer_id(skeleton_name: str) -> int:
    match = _PERF_RE.search(skeleton_name)
    if not match:
        raise ValueError(f"Cannot parse P### from skeleton name: {skeleton_name}")
    return int(match.group(1))


def index_parsing_subjects(parse_root: str) -> Dict[str, Dict]:
    subjects = {}
    for label_name, fusion_label in [("ASD", 1), ("TD", 0)]:
        label_root = os.path.join(parse_root, label_name)
        if not os.path.isdir(label_root):
            print(f"[WARN] Missing parsing folder: {label_root}")
            continue
        subdirs = [d for d in os.listdir(label_root) if os.path.isdir(os.path.join(label_root, d))]
        for sid in sorted(subdirs, key=safe_numeric_sort_key):
            color_dir = os.path.join(label_root, sid, "parsing", "colorized")
            pngs = sorted(glob.glob(os.path.join(color_dir, "*.png"))) if os.path.isdir(color_dir) else []
            subject_uid = f"{label_name}_{sid}"
            subjects[subject_uid] = {
                "subject_uid": subject_uid,
                "label_name": label_name,
                "label": fusion_label,
                "child_id": str(sid),
                "parse_color_dir": color_dir,
                "parse_frame_count": len(pngs),
                "parse_first_frame": os.path.basename(pngs[0]) if pngs else "",
                "parse_last_frame": os.path.basename(pngs[-1]) if pngs else "",
                "has_parsing_dir": os.path.isdir(color_dir),
                "has_parsing_frames": len(pngs) > 0,
            }
    return subjects


def find_gfbmd_sequences(dataset_root: str) -> List[Tuple[str, str, int]]:
    seqs = []
    autism_dir = os.path.join(dataset_root, "Autism", "children with ASD")
    if os.path.isdir(autism_dir):
        for child in sorted(os.listdir(autism_dir), key=safe_numeric_sort_key):
            if not str(child).isdigit():
                continue
            xlsx_path = os.path.join(autism_dir, str(child), "video", "8.xlsx")
            if os.path.isfile(xlsx_path):
                seqs.append((xlsx_path, "Autism", int(child)))

    typical_dir = os.path.join(dataset_root, "Typical")
    if os.path.isdir(typical_dir):
        for child in sorted(os.listdir(typical_dir), key=safe_numeric_sort_key):
            if not str(child).isdigit():
                continue
            xlsx_path = os.path.join(typical_dir, str(child), "video", "8.xlsx")
            if os.path.isfile(xlsx_path):
                seqs.append((xlsx_path, "Typical", int(child)))
    return seqs


def build_expected_subject_to_skeleton_mapping(dataset_root: str) -> Dict[str, Dict]:
    mapping = {}
    for next_pid, (xlsx_path, cls_name, child_id) in enumerate(find_gfbmd_sequences(dataset_root), start=1):
        if cls_name == "Autism":
            label_name, fusion_label, label_1based = "ASD", 1, 1
        else:
            label_name, fusion_label, label_1based = "TD", 0, 2
        subject_uid = f"{label_name}_{child_id}"
        mapping[subject_uid] = {
            "subject_uid": subject_uid,
            "gfbmd_class_name": cls_name,
            "child_id": str(child_id),
            "label_name": label_name,
            "label": fusion_label,
            "skeleton_name_expected": make_skeleton_name(next_pid, label_1based),
            "performer_id_expected": next_pid,
            "action_code_expected": label_1based,
            "source_xlsx_path": xlsx_path,
        }
    return mapping


def index_skeleton_stats(stats_dir: str, skeleton_dir: str) -> Dict[str, Dict]:
    names = load_txt_lines(os.path.join(stats_dir, "skes_available_name.txt"))
    performers = [int(x) for x in load_txt_lines(os.path.join(stats_dir, "performer.txt"))]
    labels_1based = [int(x) for x in load_txt_lines(os.path.join(stats_dir, "label.txt"))]
    if not (len(names) == len(performers) == len(labels_1based)):
        raise RuntimeError(
            f"Statistics length mismatch: names={len(names)}, performers={len(performers)}, labels={len(labels_1based)}"
        )

    stats = {}
    for name, performer_id, label_id in zip(names, performers, labels_1based):
        path = os.path.join(skeleton_dir, name + ".skeleton")
        stats[name] = {
            "skeleton_name": name,
            "skeleton_path": path,
            "skeleton_file_exists": os.path.isfile(path),
            "performer_id_stats": int(performer_id),
            "performer_id_from_name": parse_performer_id(name),
            "action_code_stats": int(label_id),
            "action_code_from_name": parse_action_code(name),
        }
    return stats


def build_manifest_rows(parse_root: str, gfbmd_root: str, skeleton_dir: str, statistics_dir: str):
    parsing_map = index_parsing_subjects(parse_root)
    expected_map = build_expected_subject_to_skeleton_mapping(gfbmd_root)
    skeleton_stats = index_skeleton_stats(statistics_dir, skeleton_dir)

    all_subject_uids = sorted(
        set(parsing_map).union(expected_map),
        key=lambda x: (0 if x.startswith("ASD_") else 1, safe_numeric_sort_key(x.split("_", 1)[1])),
    )

    rows = []
    issues = []
    for subject_uid in all_subject_uids:
        p = parsing_map.get(subject_uid, {})
        e = expected_map.get(subject_uid, {})
        expected_ske_name = e.get("skeleton_name_expected", "")
        s = skeleton_stats.get(expected_ske_name, {}) if expected_ske_name else {}

        label_name = p.get("label_name", e.get("label_name", ""))
        fusion_label = p.get("label", e.get("label", ""))
        action_expected = e.get("action_code_expected")
        action_stats = s.get("action_code_stats")

        label_ok = True
        if label_name == "ASD":
            label_ok = action_expected in (None, 1) and action_stats in (None, 1)
        elif label_name == "TD":
            label_ok = action_expected in (None, 2) and action_stats in (None, 2)

        has_parsing_dir = bool(p.get("has_parsing_dir", False))
        has_parsing_frames = bool(p.get("has_parsing_frames", False))
        has_expected_mapping = subject_uid in expected_map
        has_skeleton_stats = expected_ske_name in skeleton_stats if expected_ske_name else False
        skeleton_file_exists = bool(s.get("skeleton_file_exists", False))
        fusion_ready = (
            has_parsing_dir and has_parsing_frames and has_expected_mapping
            and has_skeleton_stats and skeleton_file_exists and label_ok
        )

        if not has_parsing_dir:
            issues.append(f"{subject_uid}: missing parse color dir")
        if has_parsing_dir and not has_parsing_frames:
            issues.append(f"{subject_uid}: parse color dir has 0 png frames")
        if not has_expected_mapping:
            issues.append(f"{subject_uid}: missing expected mapping from original GFBMD 8.xlsx")
        if has_expected_mapping and not has_skeleton_stats:
            issues.append(f"{subject_uid}: expected skeleton not found in statistics ({expected_ske_name})")
        if has_skeleton_stats and not skeleton_file_exists:
            issues.append(f"{subject_uid}: skeleton file missing on disk ({s.get('skeleton_path', '')})")
        if not label_ok:
            issues.append(f"{subject_uid}: label mismatch between parsing and skeleton")

        rows.append({
            "subject_uid": subject_uid,
            "label": fusion_label,
            "label_name": label_name,
            "child_id": p.get("child_id", e.get("child_id", "")),
            "parse_color_dir": p.get("parse_color_dir", ""),
            "parse_frame_count": int(p.get("parse_frame_count", 0)),
            "parse_first_frame": p.get("parse_first_frame", ""),
            "parse_last_frame": p.get("parse_last_frame", ""),
            "skeleton_name": expected_ske_name,
            "skeleton_path": s.get(
                "skeleton_path",
                os.path.join(skeleton_dir, expected_ske_name + ".skeleton") if expected_ske_name else "",
            ),
            "performer_id": e.get("performer_id_expected", ""),
            "action_code_1based": e.get("action_code_expected", ""),
            "source_xlsx_path": e.get("source_xlsx_path", ""),
            "has_parsing_dir": int(has_parsing_dir),
            "has_parsing_frames": int(has_parsing_frames),
            "has_expected_mapping": int(has_expected_mapping),
            "has_skeleton_in_stats": int(has_skeleton_stats),
            "skeleton_file_exists": int(skeleton_file_exists),
            "label_ok": int(label_ok),
            "fusion_ready": int(fusion_ready),
            "performer_id_stats": s.get("performer_id_stats", ""),
            "performer_id_from_name": s.get("performer_id_from_name", ""),
            "action_code_stats": s.get("action_code_stats", ""),
            "action_code_from_name": s.get("action_code_from_name", ""),
        })
    return rows, issues


def save_manifest(rows: List[Dict], issues: List[str], out_dir: str, path_summary: Dict[str, str]):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "subject_manifest.csv")
    json_path = os.path.join(out_dir, "manifest_summary.json")
    issues_path = os.path.join(out_dir, "manifest_issues.txt")

    fieldnames = [
        "subject_uid", "label", "label_name", "child_id",
        "parse_color_dir", "parse_frame_count", "parse_first_frame", "parse_last_frame",
        "skeleton_name", "skeleton_path", "performer_id", "action_code_1based", "source_xlsx_path",
        "has_parsing_dir", "has_parsing_frames", "has_expected_mapping", "has_skeleton_in_stats",
        "skeleton_file_exists", "label_ok", "fusion_ready",
        "performer_id_stats", "performer_id_from_name", "action_code_stats", "action_code_from_name",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(issues_path, "w", encoding="utf-8") as f:
        if issues:
            f.write("\n".join(issues) + "\n")
        else:
            f.write("No issues found.\n")

    labels = np.array([int(r["label"]) for r in rows if str(r["label"]) != ""], dtype=int)
    fusion_ready = np.array([int(r["fusion_ready"]) for r in rows], dtype=int)
    parse_counts = np.array([int(r["parse_frame_count"]) for r in rows], dtype=int)
    summary = {
        "paths": path_summary,
        "counts": {
            "total_subjects_in_manifest": len(rows),
            "asd_subjects": int((labels == 1).sum()) if labels.size else 0,
            "td_subjects": int((labels == 0).sum()) if labels.size else 0,
            "fusion_ready_subjects": int((fusion_ready == 1).sum()) if fusion_ready.size else 0,
            "not_ready_subjects": int((fusion_ready != 1).sum()) if fusion_ready.size else len(rows),
        },
        "parsing_frame_count_stats": {
            "min": int(parse_counts.min()) if len(parse_counts) else 0,
            "max": int(parse_counts.max()) if len(parse_counts) else 0,
            "mean": float(parse_counts.mean()) if len(parse_counts) else 0.0,
        },
        "files": {
            "subject_manifest_csv": csv_path,
            "manifest_issues_txt": issues_path,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return csv_path, json_path, issues_path, summary


def main(args) -> None:
    rows, issues = build_manifest_rows(
        args.parse_root,
        args.gfbmd_root,
        args.skeleton_dir,
        args.statistics_dir,
    )
    paths = {
        "PARSE_ROOT": args.parse_root,
        "GFBMD_DATASET_ROOT": args.gfbmd_root,
        "SKELETON_DIR": args.skeleton_dir,
        "STATISTICS_DIR": args.statistics_dir,
        "OUT_DIR": args.out_dir,
    }
    csv_path, json_path, issues_path, summary = save_manifest(rows, issues, args.out_dir, paths)
    print(f"Manifest: {csv_path}")
    print(f"Summary: {json_path}")
    print(f"Issues: {issues_path} ({len(issues)})")
    print(json.dumps(summary["counts"], indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parse-root", required=True, help="Root containing ASD/<id>/parsing and TD/<id>/parsing")
    parser.add_argument("--gfbmd-root", required=True, help="Original GFBMD dataset root")
    parser.add_argument("--skeleton-dir", required=True, help="Directory containing .skeleton files")
    parser.add_argument("--statistics-dir", required=True, help="Directory containing statistics/*.txt")
    parser.add_argument("--out-dir", required=True, help="Output directory for manifest files")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
