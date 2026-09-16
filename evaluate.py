"""Evaluate saved DeCAPS-Net predictions and export metrics plus ROC curve data.

ASDPose:
    python evaluate.py --dataset asdpose --work-dir work_dir/asdpose_skeleton --out-dir work_dir/asdpose_skeleton/evaluation

GFBMD fusion:
    python evaluate.py --dataset gfbmd --work-dir work_dir/gfbmd_fusion --out-dir work_dir/gfbmd_fusion/evaluation
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_KEYS = ["acc", "precision", "recall", "f1", "auc", "ap"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)


def check_binary_labels(y_true, source):
    labels = np.unique(y_true)
    if not np.array_equal(labels, np.array([0, 1])):
        raise ValueError(f"{source}: ROC evaluation requires both labels 0 and 1; got {labels.tolist()}")


def metrics_at_threshold(y_true, prob, threshold):
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auc = float("nan")
    ap = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, prob))
        ap = float(average_precision_score(y_true, prob))

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": auc,
        "ap": ap,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def clean_metrics(metrics):
    return {key: json_value(value) for key, value in metrics.items()}


def roc_rows(series, y_true, prob):
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    return [
        {"series": series, "fpr": float(f), "tpr": float(t), "threshold": float(th)}
        for f, t, th in zip(fpr, tpr, thresholds)
    ]


def write_roc_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["series", "fpr", "tpr", "threshold"])
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(rows, path):
    fieldnames = ["series"] + list(rows[0]["metrics"].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {"series": row["series"]}
            out.update({key: json_value(value) for key, value in row["metrics"].items()})
            writer.writerow(out)


def load_asdpose_predictions(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError(f"Prediction CSV is empty: {path}")
    if "y_true" not in fieldnames:
        raise ValueError(f"Missing y_true column: {path}")

    y_true = np.asarray([int(float(row["y_true"])) for row in rows], dtype=np.int64)
    if "prob_1" in fieldnames and all(row.get("prob_1", "") != "" for row in rows):
        prob = np.asarray([float(row["prob_1"]) for row in rows], dtype=np.float64)
    elif {"logit_0", "logit_1"}.issubset(fieldnames):
        logits = np.asarray([[float(row["logit_0"]), float(row["logit_1"])] for row in rows], dtype=np.float64)
        prob = softmax_np(logits)[:, 1]
    else:
        raise ValueError(f"Need prob_1 or logit_0/logit_1 columns: {path}")

    identifiers = [row.get("identifier", str(i)) for i, row in enumerate(rows)]
    return y_true, prob, identifiers


def evaluate_asdpose(predictions_path, out_dir, thresholds):
    y_true, prob, _ = load_asdpose_predictions(predictions_path)
    check_binary_labels(y_true, predictions_path)
    ensure_dir(out_dir)

    threshold_metrics = [clean_metrics(metrics_at_threshold(y_true, prob, threshold)) for threshold in thresholds]
    primary = threshold_metrics[0]
    roc_path = os.path.join(out_dir, "roc.csv")
    metrics_csv_path = os.path.join(out_dir, "metrics.csv")
    metrics_json_path = os.path.join(out_dir, "metrics.json")

    write_roc_csv(roc_rows("test", y_true, prob), roc_path)
    write_metrics_csv([{"series": "test", "metrics": m} for m in threshold_metrics], metrics_csv_path)

    payload = {
        "dataset": "asdpose",
        "predictions": predictions_path,
        "num_samples": int(len(y_true)),
        "thresholds": [float(x) for x in thresholds],
        "metrics": primary,
        "threshold_metrics": threshold_metrics,
        "outputs": {"metrics_csv": metrics_csv_path, "roc_csv": roc_path},
    }
    save_json(payload, metrics_json_path)

    print(f"ASDPose n={len(y_true)} acc={primary['acc']:.4f} f1={primary['f1']:.4f} auc={primary['auc']:.4f}")
    print(f"Metrics: {metrics_json_path}")
    print(f"ROC: {roc_path}")


def parse_fold_from_path(path):
    match = re.search(r"fold(\d+)", path.replace("\\", "/"))
    if match is None:
        raise ValueError(f"Cannot parse fold number from: {path}")
    return int(match.group(1))


def resolve_score_paths(score_pattern, num_folds=None):
    paths = glob.glob(score_pattern)
    if not paths:
        raise FileNotFoundError(f"No score files matched: {score_pattern}")

    by_fold = defaultdict(list)
    for path in paths:
        by_fold[parse_fold_from_path(path)].append(path)

    fold_ids = sorted(by_fold)
    if num_folds is not None:
        expected = list(range(1, num_folds + 1))
        missing = [fold for fold in expected if fold not in by_fold]
        if missing:
            raise FileNotFoundError(f"Missing folds {missing} for pattern: {score_pattern}")
        fold_ids = expected

    return [max(by_fold[fold], key=os.path.getmtime) for fold in fold_ids]


def load_score_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a score dictionary: {path}")
    y_true = np.asarray(data["y_true"], dtype=np.int64)
    if "prob" in data:
        prob = np.asarray(data["prob"], dtype=np.float64)
    else:
        logits = np.asarray(data["logits"], dtype=np.float64)
        prob = 1.0 / (1.0 + np.exp(-logits))
    return y_true, prob


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1:
        return {"mean": float(values.mean()), "std": 0.0}
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def evaluate_gfbmd(score_pattern, out_dir, thresholds, num_folds):
    score_paths = resolve_score_paths(score_pattern, num_folds)
    ensure_dir(out_dir)

    fold_entries = []
    fold_metrics_primary = []
    metrics_rows = []
    roc_output_rows = []
    all_y_true = []
    all_prob = []

    for path in score_paths:
        fold = parse_fold_from_path(path)
        y_true, prob = load_score_pkl(path)
        check_binary_labels(y_true, path)
        threshold_metrics = [clean_metrics(metrics_at_threshold(y_true, prob, threshold)) for threshold in thresholds]
        primary = threshold_metrics[0]
        fold_metrics_primary.append(primary)
        fold_entries.append({
            "fold": int(fold),
            "n": int(len(y_true)),
            "score_file": path,
            "metrics": primary,
            "threshold_metrics": threshold_metrics,
        })
        for metrics in threshold_metrics:
            metrics_rows.append({"series": f"fold_{fold}", "metrics": metrics})
        roc_output_rows.extend(roc_rows(f"fold_{fold}", y_true, prob))
        all_y_true.append(y_true)
        all_prob.append(prob)

    y_true = np.concatenate(all_y_true)
    prob = np.concatenate(all_prob)
    pooled_threshold_metrics = [clean_metrics(metrics_at_threshold(y_true, prob, threshold)) for threshold in thresholds]
    for metrics in pooled_threshold_metrics:
        metrics_rows.append({"series": "pooled", "metrics": metrics})
    roc_output_rows.extend(roc_rows("pooled", y_true, prob))

    summary = {}
    for key in METRIC_KEYS:
        summary[key] = mean_std([metrics[key] for metrics in fold_metrics_primary])

    metrics_csv_path = os.path.join(out_dir, "metrics.csv")
    roc_path = os.path.join(out_dir, "roc.csv")
    metrics_json_path = os.path.join(out_dir, "metrics.json")
    write_metrics_csv(metrics_rows, metrics_csv_path)
    write_roc_csv(roc_output_rows, roc_path)

    payload = {
        "dataset": "gfbmd",
        "score_pattern": score_pattern,
        "num_folds": int(len(score_paths)),
        "num_samples": int(len(y_true)),
        "thresholds": [float(x) for x in thresholds],
        "folds": fold_entries,
        "summary": {
            "primary_threshold": float(thresholds[0]),
            **{key: json_value(value) for key, value in summary.items()},
        },
        "pooled": {
            "n": int(len(y_true)),
            "metrics": pooled_threshold_metrics[0],
            "threshold_metrics": pooled_threshold_metrics,
        },
        "outputs": {"metrics_csv": metrics_csv_path, "roc_csv": roc_path},
    }
    save_json(payload, metrics_json_path)

    print(
        f"GFBMD folds={len(score_paths)} pooled_n={len(y_true)} "
        f"acc={summary['acc']['mean']:.4f}±{summary['acc']['std']:.4f} "
        f"f1={summary['f1']['mean']:.4f}±{summary['f1']['std']:.4f} "
        f"auc={summary['auc']['mean']:.4f}±{summary['auc']['std']:.4f}"
    )
    print(f"Metrics: {metrics_json_path}")
    print(f"ROC: {roc_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["asdpose", "gfbmd"])
    parser.add_argument("--work-dir", default=None, help="Training work directory used to resolve default prediction files")
    parser.add_argument("--predictions", default=None, help="ASDPose test_predictions.csv")
    parser.add_argument("--score-pattern", default=None, help="GFBMD fold score glob, e.g. 'work_dir/fold*/val_best_scores.pkl'")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    parser.add_argument("--num-folds", type=int, default=None, help="Optional expected fold count for GFBMD")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset == "asdpose":
        predictions = args.predictions
        if predictions is None:
            if args.work_dir is None:
                raise ValueError("ASDPose evaluation needs --predictions or --work-dir.")
            predictions = os.path.join(args.work_dir, "test_predictions.csv")
        if not os.path.isfile(predictions):
            raise FileNotFoundError(f"Prediction CSV not found: {predictions}")
        evaluate_asdpose(predictions, args.out_dir, args.thresholds)
    else:
        score_pattern = args.score_pattern
        if score_pattern is None:
            if args.work_dir is None:
                raise ValueError("GFBMD evaluation needs --score-pattern or --work-dir.")
            score_pattern = os.path.join(args.work_dir, "fold*", "val_best_scores.pkl")
        evaluate_gfbmd(score_pattern, args.out_dir, args.thresholds, args.num_folds)


if __name__ == "__main__":
    main()
