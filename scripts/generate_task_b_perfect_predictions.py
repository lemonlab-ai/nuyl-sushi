import argparse
import json
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GT_JSON = ROOT_DIR / "artifacts" / "task_b" / "activitynet_fine.json"
DEFAULT_OUT_JSON = ROOT_DIR / "artifacts" / "task_b" / "pred_perfect.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate perfect Task B prediction JSON from ActivityNet-style GT."
    )
    parser.add_argument("--gt-json", default=str(DEFAULT_GT_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--score", type=float, default=1.0, help="Score assigned to each copied GT segment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt_path = Path(args.gt_json)
    out_path = Path(args.output_json)
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    database = payload.get("database", {})
    if not isinstance(database, dict):
        raise RuntimeError("Invalid GT JSON: database must be a dict.")

    results: Dict[str, List[Dict]] = {}
    num_segments = 0
    for video_id, item in database.items():
        anns = item.get("annotations", [])
        out_list = []
        if isinstance(anns, list):
            for ann in anns:
                seg = ann.get("segment", [0.0, 0.0])
                label = ann.get("label", "")
                if not isinstance(seg, list) or len(seg) < 2:
                    continue
                out_list.append(
                    {
                        "segment": [float(seg[0]), float(seg[1])],
                        "label": str(label),
                        "score": float(args.score),
                    }
                )
                num_segments += 1
        results[str(video_id)] = out_list

    out_payload = {"results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[DONE] Generated perfect Task B prediction JSON.")
    print(
        json.dumps(
            {
                "gt_json": str(gt_path.resolve()),
                "output_json": str(out_path.resolve()),
                "num_videos": len(results),
                "num_segments": num_segments,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

