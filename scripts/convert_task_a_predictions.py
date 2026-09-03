import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_MAP = ROOT_DIR / "artifacts" / "task_a" / "class_map_fine.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert model Task A predictions to eval CSV format.")
    parser.add_argument("--input", required=True, help="Input prediction file (csv/json).")
    parser.add_argument("--output-csv", required=True, help="Output CSV: clip_id,pred_label_id")
    parser.add_argument("--class-map", default=str(DEFAULT_CLASS_MAP), help="Label->id map JSON.")
    parser.add_argument("--format", choices=["auto", "csv", "json"], default="auto")
    return parser.parse_args()


def load_class_map(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(k): int(v) for k, v in payload.items()}


def infer_format(path: Path, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    ext = path.suffix.lower()
    if ext in {".csv", ".tsv"}:
        return "csv"
    return "json"


def resolve_pred_id(row: Dict, class_map: Dict[str, int]) -> int:
    for key in ("pred_label_id", "label_id", "pred_id", "class_id"):
        if key in row and str(row[key]).strip() != "":
            return int(float(row[key]))
    for key in ("pred_label", "label", "class_name"):
        if key in row and str(row[key]).strip() != "":
            label = str(row[key]).strip()
            if label in class_map:
                return int(class_map[label])
    if "topk_ids" in row and str(row["topk_ids"]).strip():
        first = str(row["topk_ids"]).strip().split()[0]
        return int(float(first))
    if "topk_labels" in row and str(row["topk_labels"]).strip():
        first = str(row["topk_labels"]).strip().split()[0]
        if first in class_map:
            return int(class_map[first])
    raise ValueError(f"Cannot resolve pred_label_id from row: {row}")


def convert_csv(path: Path, class_map: Dict[str, int]) -> List[Dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        clip_id = str(r.get("clip_id", "")).strip()
        if not clip_id:
            continue
        pred_id = resolve_pred_id(r, class_map)
        out.append({"clip_id": clip_id, "pred_label_id": pred_id})
    return out


def normalize_json_entries(payload) -> List[Dict]:
    if isinstance(payload, dict):
        if "predictions" in payload and isinstance(payload["predictions"], list):
            return payload["predictions"]
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
        # map form: {clip_id: pred}
        entries = []
        for k, v in payload.items():
            if isinstance(v, dict):
                item = dict(v)
                item.setdefault("clip_id", k)
                entries.append(item)
            else:
                entries.append({"clip_id": k, "pred_label_id": v})
        return entries
    if isinstance(payload, list):
        return payload
    return []


def convert_json(path: Path, class_map: Dict[str, int]) -> List[Dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = normalize_json_entries(payload)
    out = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        clip_id = str(e.get("clip_id", "")).strip()
        if not clip_id:
            continue
        pred_id = resolve_pred_id(e, class_map)
        out.append({"clip_id": clip_id, "pred_label_id": pred_id})
    return out


def write_output(path: Path, rows: List[Dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "pred_label_id"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    src = Path(args.input)
    if not src.exists():
        raise FileNotFoundError(src)
    fmt = infer_format(src, args.format)
    class_map = load_class_map(Path(args.class_map))

    if fmt == "csv":
        rows = convert_csv(src, class_map)
    else:
        rows = convert_json(src, class_map)
    if not rows:
        raise RuntimeError(f"No valid prediction rows found in {src}")

    out = Path(args.output_csv)
    write_output(out, rows)
    print(f"[DONE] Converted Task A predictions: {out}")
    print(json.dumps({"num_rows": len(rows), "format": fmt}, indent=2))


if __name__ == "__main__":
    main()
