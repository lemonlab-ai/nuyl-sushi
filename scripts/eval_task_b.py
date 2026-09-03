import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GT_JSON = ROOT_DIR / "artifacts" / "task_b" / "activitynet_fine.json"
DEFAULT_IOU = "0.3,0.5,0.75"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task B temporal localization predictions.")
    parser.add_argument("--gt-json", default=str(DEFAULT_GT_JSON), help="Ground-truth ActivityNet-style JSON.")
    parser.add_argument("--pred-json", required=True, help="Prediction JSON.")
    parser.add_argument("--iou-thresholds", default=DEFAULT_IOU, help="Comma-separated IoU thresholds.")
    parser.add_argument("--subset", choices=["training", "validation", "testing", "all"], default="all")
    parser.add_argument("--output-json", default="", help="Optional path to save metrics JSON.")
    return parser.parse_args()


def segment_iou(seg1: List[float], seg2: List[float]) -> float:
    s1, e1 = float(seg1[0]), float(seg1[1])
    s2, e2 = float(seg2[0]), float(seg2[1])
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    if union <= 0:
        return 0.0
    return inter / union


def parse_gt(gt_payload: Dict, subset: str) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for vid, info in gt_payload.get("database", {}).items():
        if subset != "all" and info.get("subset") != subset:
            continue
        for ann in info.get("annotations", []):
            out[ann["label"]].append({"video_id": vid, "segment": ann["segment"]})
    return out


def parse_pred(pred_payload: Dict) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    if "results" in pred_payload and isinstance(pred_payload["results"], dict):
        for vid, preds in pred_payload["results"].items():
            for p in preds:
                out[p["label"]].append(
                    {
                        "video_id": vid,
                        "segment": p["segment"],
                        "score": float(p.get("score", 0.0)),
                    }
                )
        return out

    if isinstance(pred_payload.get("predictions"), list):
        entries = pred_payload["predictions"]
    elif isinstance(pred_payload.get("results"), list):
        entries = pred_payload["results"]
    else:
        entries = pred_payload if isinstance(pred_payload, list) else []

    for p in entries:
        out[p["label"]].append(
            {
                "video_id": p["video_id"],
                "segment": p["segment"],
                "score": float(p.get("score", 0.0)),
            }
        )
    return out


def average_precision(rec: List[float], prec: List[float]) -> float:
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_class_at_iou(gt_cls: List[Dict], pred_cls: List[Dict], iou_thr: float) -> Tuple[float, float]:
    if not gt_cls:
        return 0.0, 0.0
    gt_by_vid = defaultdict(list)
    for g in gt_cls:
        gt_by_vid[g["video_id"]].append(g["segment"])
    matched = {vid: [False] * len(segs) for vid, segs in gt_by_vid.items()}

    preds = sorted(pred_cls, key=lambda x: x["score"], reverse=True)
    tp, fp = [], []
    for p in preds:
        vid = p["video_id"]
        pseg = p["segment"]
        best_iou = 0.0
        best_idx = -1
        for i, gseg in enumerate(gt_by_vid.get(vid, [])):
            if matched[vid][i]:
                continue
            iou = segment_iou(pseg, gseg)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_thr and best_idx >= 0:
            matched[vid][best_idx] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not tp:
        return 0.0, 0.0

    tp_cum, fp_cum = [], []
    ctp, cfp = 0.0, 0.0
    for a, b in zip(tp, fp):
        ctp += a
        cfp += b
        tp_cum.append(ctp)
        fp_cum.append(cfp)

    n_gt = len(gt_cls)
    rec = [x / n_gt for x in tp_cum]
    prec = [tp_cum[i] / max(tp_cum[i] + fp_cum[i], 1e-12) for i in range(len(tp_cum))]
    ap = average_precision(rec, prec)
    recall = rec[-1] if rec else 0.0
    return ap, recall


def main() -> None:
    args = parse_args()
    iou_thresholds = [float(x.strip()) for x in args.iou_thresholds.split(",") if x.strip()]

    gt_payload = json.loads(Path(args.gt_json).read_text(encoding="utf-8"))
    pred_payload = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))

    gt_by_class = parse_gt(gt_payload, subset=args.subset)
    pred_by_class = parse_pred(pred_payload)
    classes = sorted(set(gt_by_class.keys()) | set(pred_by_class.keys()))

    ap_by_iou = {}
    ap_detail = {}
    recall_by_iou = {}
    recall_detail = {}
    for thr in iou_thresholds:
        cls_aps = []
        cls_recalls = []
        per_cls = {}
        per_cls_recall = {}
        for c in classes:
            ap, recall = evaluate_class_at_iou(gt_by_class[c], pred_by_class.get(c, []), thr)
            cls_aps.append(ap)
            cls_recalls.append(recall)
            per_cls[c] = ap
            per_cls_recall[c] = recall
        map_thr = sum(cls_aps) / len(cls_aps) if cls_aps else 0.0
        r_thr = sum(cls_recalls) / len(cls_recalls) if cls_recalls else 0.0
        ap_by_iou[f"{thr:.2f}"] = map_thr
        ap_detail[f"{thr:.2f}"] = per_cls
        recall_by_iou[f"{thr:.2f}"] = r_thr
        recall_detail[f"{thr:.2f}"] = per_cls_recall

    avg_map = sum(ap_by_iou.values()) / len(ap_by_iou) if ap_by_iou else 0.0
    avg_recall = sum(recall_by_iou.values()) / len(recall_by_iou) if recall_by_iou else 0.0
    metrics = {
        "subset": args.subset,
        "num_classes": len(classes),
        "num_gt_instances": sum(len(v) for v in gt_by_class.values()),
        "mAP_by_IoU": ap_by_iou,
        "average_mAP": avg_map,
        "per_class_AP_by_IoU": ap_detail,
        "Recall_by_IoU": recall_by_iou,
        "average_Recall": avg_recall,
        "per_class_Recall_by_IoU": recall_detail,
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
