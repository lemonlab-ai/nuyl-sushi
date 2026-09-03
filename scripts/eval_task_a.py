import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT_DIR / "artifacts" / "task_a" / "task_a_fine_manifest.csv"
DEFAULT_CLASS_MAP = ROOT_DIR / "artifacts" / "task_a" / "class_map_fine.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task A classification predictions.")
    parser.add_argument("--manifest-csv", default=str(DEFAULT_MANIFEST), help="Ground-truth manifest CSV.")
    parser.add_argument("--pred-csv", required=True, help="Prediction CSV.")
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP), help="Label->id map JSON.")
    parser.add_argument("--subset", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--output-json", default="", help="Optional path to save metrics JSON.")
    parser.add_argument("--per-class-csv", default="", help="Optional output CSV for per-class metrics.")
    parser.add_argument("--confusion-csv", default="", help="Optional output CSV for confusion matrix.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_class_map(path: Path) -> Dict[str, int]:
    return {k: int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def resolve_pred_label_id(
    row: Dict[str, str], class_map: Dict[str, int], inv_class_map: Dict[int, str]
) -> int:
    if "pred_label_id" in row and row["pred_label_id"] != "":
        return int(row["pred_label_id"])
    if "pred_label" in row and row["pred_label"] != "":
        label = row["pred_label"]
        if label in class_map:
            return class_map[label]
    if "topk_ids" in row and row["topk_ids"].strip():
        return int(row["topk_ids"].split()[0])
    if "topk_labels" in row and row["topk_labels"].strip():
        label = row["topk_labels"].split()[0]
        if label in class_map:
            return class_map[label]
    raise ValueError(f"Cannot resolve predicted label id for row: {row}")


def build_confusion_matrix(
    y_true: List[int], y_pred: List[int], num_classes: int
) -> Tuple[List[List[int]], List[int]]:
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    missing_pred_by_true = [0 for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        if not (0 <= t < num_classes):
            continue
        if 0 <= p < num_classes:
            cm[t][p] += 1
        else:
            missing_pred_by_true[t] += 1
    return cm, missing_pred_by_true


def macro_f1_and_balanced_acc(cm: List[List[int]], missing_pred_by_true: List[int]) -> Tuple[float, float]:
    num_classes = len(cm)
    f1_scores = []
    recalls = []
    for i in range(num_classes):
        tp = cm[i][i]
        fn = sum(cm[i][j] for j in range(num_classes) if j != i) + int(missing_pred_by_true[i])
        fp = sum(cm[j][i] for j in range(num_classes) if j != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
        recalls.append(recall)
    macro_f1 = sum(f1_scores) / num_classes if num_classes else 0.0
    balanced_acc = sum(recalls) / num_classes if num_classes else 0.0
    return macro_f1, balanced_acc


def per_class_metrics(
    cm: List[List[int]],
    missing_pred_by_true: List[int],
    inv_map: Dict[int, str],
) -> Dict[str, Dict[str, float]]:
    num_classes = len(cm)
    total = sum(sum(r) for r in cm) + sum(int(v) for v in missing_pred_by_true)
    out: Dict[str, Dict[str, float]] = {}
    for i in range(num_classes):
        label = inv_map.get(i, f"__ID_{i}")
        tp = cm[i][i]
        fn = sum(cm[i][j] for j in range(num_classes) if j != i) + int(missing_pred_by_true[i])
        fp = sum(cm[j][i] for j in range(num_classes) if j != i)
        tn = total - tp - fn - fp
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        out[label] = {
            "label_id": i,
            "support": float(support),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
        }
    return out


def write_per_class_csv(path: Path, per_class: Dict[str, Dict[str, float]]) -> None:
    rows = []
    for label, m in per_class.items():
        rows.append(
            {
                "label": label,
                "label_id": int(m["label_id"]),
                "support": int(m["support"]),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "specificity": m["specificity"],
            }
        )
    rows.sort(key=lambda r: r["label_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "label_id", "support", "precision", "recall", "f1", "specificity"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(path: Path, cm: List[List[int]], inv_map: Dict[int, str]) -> None:
    num_classes = len(cm)
    labels = [inv_map.get(i, f"__ID_{i}") for i in range(num_classes)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for i in range(num_classes):
            writer.writerow([labels[i]] + cm[i])


def main() -> None:
    args = parse_args()
    manifest_rows = read_csv_rows(Path(args.manifest_csv))
    pred_rows = read_csv_rows(Path(args.pred_csv))
    class_map = load_class_map(Path(args.class_map))
    inv_map = {v: k for k, v in class_map.items()}

    gt_by_clip = {}
    for row in manifest_rows:
        subset = row.get("subset", "train")
        if args.subset != "all" and subset != args.subset:
            continue
        gt_by_clip[row["clip_id"]] = int(row["label_id"])

    pred_by_clip = {}
    for row in pred_rows:
        clip_id = row.get("clip_id", "")
        if not clip_id:
            continue
        pred_by_clip[clip_id] = resolve_pred_label_id(row, class_map, inv_map)

    keys = sorted(gt_by_clip.keys())
    if not keys:
        raise ValueError("No GT clip found after subset filtering.")

    y_true = [gt_by_clip[k] for k in keys]
    y_pred = [pred_by_clip.get(k, -1) for k in keys]
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    top1 = correct / total if total else 0.0

    cm, missing_pred_by_true = build_confusion_matrix(y_true, y_pred, num_classes=len(class_map))
    macro_f1, bal_acc = macro_f1_and_balanced_acc(cm, missing_pred_by_true)
    per_class = per_class_metrics(cm, missing_pred_by_true, inv_map)

    per_class_support = defaultdict(int)
    for t in y_true:
        per_class_support[t] += 1

    metrics = {
        "num_samples": total,
        "num_gt_clips": len(gt_by_clip),
        "num_pred_clips": len(pred_by_clip),
        "num_overlap_clips": len(set(keys) & set(pred_by_clip.keys())),
        "coverage": (len(set(keys) & set(pred_by_clip.keys())) / len(gt_by_clip)) if gt_by_clip else 0.0,
        "num_missing_pred_clips": int(sum(1 for p in y_pred if p < 0)),
        "top1_accuracy": top1,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "subset": args.subset,
        "num_classes": len(class_map),
        "per_class_support": {inv_map[k]: v for k, v in sorted(per_class_support.items())},
        "per_class_metrics": per_class,
        "confusion_matrix": cm,
        "missing_pred_by_true_class": {inv_map[i]: int(v) for i, v in enumerate(missing_pred_by_true)},
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.per_class_csv:
        write_per_class_csv(Path(args.per_class_csv), per_class)
    if args.confusion_csv:
        write_confusion_csv(Path(args.confusion_csv), cm, inv_map)


if __name__ == "__main__":
    main()
