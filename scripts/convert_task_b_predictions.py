import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert model Task B predictions to eval JSON format.")
    parser.add_argument("--input", required=True, help="Input prediction file (csv/json).")
    parser.add_argument("--output-json", required=True, help="Output JSON in {'results': {video_id: [...]}} format.")
    parser.add_argument("--format", choices=["auto", "csv", "json", "pkl"], default="auto")
    parser.add_argument(
        "--label-map",
        default="",
        help="Optional label map JSON (string->id) for converting numeric label ids to label names.",
    )
    return parser.parse_args()


def infer_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    if path.suffix.lower() in {".csv", ".tsv"}:
        return "csv"
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return "pkl"
    return "json"


def to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def parse_label_id(raw) -> int | None:
    try:
        return int(raw)
    except Exception:
        try:
            return int(float(raw))
        except Exception:
            return None


def load_label_map(path: Path) -> Dict[int, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    id_to_label: Dict[int, str] = {}
    if isinstance(payload, dict):
        for label, idx in payload.items():
            label_id = parse_label_id(idx)
            if label_id is None:
                continue
            id_to_label[label_id] = str(label)
    return id_to_label


def normalize_label(raw_label, id_to_label: Dict[int, str]) -> str:
    label_id = parse_label_id(raw_label)
    if label_id is not None and label_id in id_to_label:
        return id_to_label[label_id]
    return str(raw_label).strip()


def csv_to_results(path: Path, id_to_label: Dict[int, str]) -> Dict[str, List[Dict]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = defaultdict(list)
    for r in rows:
        vid = str(r.get("video_id", "")).strip()
        if not vid:
            continue
        label = normalize_label(r.get("label", r.get("class_name", "")), id_to_label)
        if not label:
            continue
        score = to_float(r.get("score", 0.0))
        if "segment" in r and str(r["segment"]).strip():
            try:
                seg = json.loads(str(r["segment"]))
                start, end = float(seg[0]), float(seg[1])
            except Exception:
                continue
        else:
            start = to_float(r.get("start", r.get("start_sec", 0.0)))
            end = to_float(r.get("end", r.get("end_sec", 0.0)))
        out[vid].append({"segment": [start, end], "label": label, "score": score})
    return dict(out)


def pkl_to_results(path: Path, id_to_label: Dict[int, str]) -> Dict[str, List[Dict]]:
    with path.open("rb") as f:
        payload = pickle.load(f)

    out = defaultdict(list)
    if isinstance(payload, dict) and {"video-id", "t-start", "t-end", "label", "score"}.issubset(payload.keys()):
        vids = payload["video-id"]
        starts = payload["t-start"]
        ends = payload["t-end"]
        labels = payload["label"]
        scores = payload["score"]
        for vid, start, end, label, score in zip(vids, starts, ends, labels, scores):
            vid_s = str(vid).strip()
            label_s = normalize_label(label, id_to_label)
            if not vid_s or not label_s:
                continue
            out[vid_s].append(
                {
                    "segment": [to_float(start), to_float(end)],
                    "label": label_s,
                    "score": to_float(score),
                }
            )
        return dict(out)

    if isinstance(payload, dict):
        for vid, preds in payload.items():
            if not isinstance(preds, list):
                continue
            for p in preds:
                if not isinstance(p, dict):
                    continue
                label = normalize_label(p.get("label", ""), id_to_label)
                seg = p.get("segment", [p.get("start", 0.0), p.get("end", 0.0)])
                if not label or not isinstance(seg, (list, tuple)) or len(seg) < 2:
                    continue
                out[str(vid)].append(
                    {
                        "segment": [to_float(seg[0]), to_float(seg[1])],
                        "label": label,
                        "score": to_float(p.get("score", 0.0)),
                    }
                )
        return dict(out)

    return {}


def normalize_json_payload(payload, id_to_label: Dict[int, str]) -> Dict[str, List[Dict]]:
    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], dict):
            out = defaultdict(list)
            for vid, preds in payload["results"].items():
                if not isinstance(preds, list):
                    continue
                for p in preds:
                    if not isinstance(p, dict):
                        continue
                    seg = p.get("segment", [p.get("start", 0.0), p.get("end", 0.0)])
                    label = normalize_label(p.get("label", ""), id_to_label)
                    if not label or not isinstance(seg, (list, tuple)) or len(seg) < 2:
                        continue
                    out[str(vid)].append(
                        {
                            "segment": [to_float(seg[0]), to_float(seg[1])],
                            "label": label,
                            "score": to_float(p.get("score", 0.0)),
                        }
                    )
            return dict(out)
        if "predictions" in payload and isinstance(payload["predictions"], list):
            entries = payload["predictions"]
        elif "results" in payload and isinstance(payload["results"], list):
            entries = payload["results"]
        else:
            # maybe dict keyed by video id -> list
            if all(isinstance(v, list) for v in payload.values()):
                out = defaultdict(list)
                for vid, preds in payload.items():
                    for p in preds:
                        if not isinstance(p, dict):
                            continue
                        seg = p.get("segment", [p.get("start", 0.0), p.get("end", 0.0)])
                        label = normalize_label(p.get("label", ""), id_to_label)
                        if not label or not isinstance(seg, (list, tuple)) or len(seg) < 2:
                            continue
                        out[str(vid)].append(
                            {
                                "segment": [to_float(seg[0]), to_float(seg[1])],
                                "label": label,
                                "score": to_float(p.get("score", 0.0)),
                            }
                        )
                return dict(out)
            entries = []
        out = defaultdict(list)
        for e in entries:
            if not isinstance(e, dict):
                continue
            vid = str(e.get("video_id", "")).strip()
            label = normalize_label(e.get("label", ""), id_to_label)
            if not vid or not label:
                continue
            seg = e.get("segment", [e.get("start", 0.0), e.get("end", 0.0)])
            if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                continue
            out[vid].append(
                {
                    "segment": [to_float(seg[0]), to_float(seg[1])],
                    "label": label,
                    "score": to_float(e.get("score", 0.0)),
                }
            )
        return dict(out)

    if isinstance(payload, list):
        out = defaultdict(list)
        for e in payload:
            if not isinstance(e, dict):
                continue
            vid = str(e.get("video_id", "")).strip()
            label = normalize_label(e.get("label", ""), id_to_label)
            if not vid or not label:
                continue
            seg = e.get("segment", [e.get("start", 0.0), e.get("end", 0.0)])
            if not isinstance(seg, (list, tuple)) or len(seg) < 2:
                continue
            out[vid].append(
                {
                    "segment": [to_float(seg[0]), to_float(seg[1])],
                    "label": label,
                    "score": to_float(e.get("score", 0.0)),
                }
            )
        return dict(out)
    return {}


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)
    id_to_label: Dict[int, str] = {}
    if args.label_map.strip():
        id_to_label = load_label_map(Path(args.label_map))
    fmt = infer_format(src, args.format)
    if fmt == "csv":
        results = csv_to_results(src, id_to_label)
    elif fmt == "pkl":
        results = pkl_to_results(src, id_to_label)
    else:
        payload = json.loads(src.read_text(encoding="utf-8"))
        results = normalize_json_payload(payload, id_to_label)
    if not results:
        raise RuntimeError(f"No valid Task B predictions found in {src}")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {"results": results}
    out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Converted Task B predictions: {out}")
    print(json.dumps({"num_videos": len(results), "format": fmt}, indent=2))


if __name__ == "__main__":
    main()
