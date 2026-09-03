import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT_DIR / "scripts"


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT_DIR)


def write_task_a_perfect_preds(manifest_csv: Path, pred_csv: Path) -> None:
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "pred_label_id"])
        writer.writeheader()
        for r in rows:
            writer.writerow({"clip_id": r["clip_id"], "pred_label_id": r["label_id"]})


def write_task_b_perfect_preds(gt_json: Path, pred_json: Path) -> None:
    payload = json.loads(gt_json.read_text(encoding="utf-8"))
    results = {}
    for vid, info in payload.get("database", {}).items():
        results[vid] = []
        for ann in info.get("annotations", []):
            results[vid].append(
                {
                    "segment": ann["segment"],
                    "label": ann["label"],
                    "score": 1.0,
                }
            )
    pred_json.parent.mkdir(parents=True, exist_ok=True)
    pred_json.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.stat().st_size == 0:
        raise RuntimeError(f"Empty file: {path}")


def main() -> None:
    py = sys.executable

    run([py, str(SCRIPTS / "generate_mock_dataset.py"), "--num-videos", "8", "--segments-per-video", "3"])
    run([py, str(SCRIPTS / "convert_via_to_master.py")])
    run([py, str(SCRIPTS / "validate_master_dataset.py")])
    run([py, str(SCRIPTS / "export_task_a_clips.py")])
    run([py, str(SCRIPTS / "export_task_b_activitynet.py")])

    master_json = ROOT_DIR / "artifacts" / "master" / "master_annotations.json"
    task_a_csv = ROOT_DIR / "artifacts" / "task_a" / "task_a_fine_manifest.csv"
    task_b_json = ROOT_DIR / "artifacts" / "task_b" / "activitynet_fine.json"
    ensure_file(master_json)
    ensure_file(task_a_csv)
    ensure_file(task_b_json)

    pred_task_a = ROOT_DIR / "artifacts" / "task_a" / "mock_pred_fine.csv"
    write_task_a_perfect_preds(task_a_csv, pred_task_a)
    run(
        [
            py,
            str(SCRIPTS / "eval_task_a.py"),
            "--manifest-csv",
            str(task_a_csv),
            "--class-map",
            str(ROOT_DIR / "artifacts" / "task_a" / "class_map_fine.json"),
            "--pred-csv",
            str(pred_task_a),
            "--subset",
            "all",
        ]
    )

    pred_task_b = ROOT_DIR / "artifacts" / "task_b" / "mock_pred_fine.json"
    write_task_b_perfect_preds(task_b_json, pred_task_b)
    run(
        [
            py,
            str(SCRIPTS / "eval_task_b.py"),
            "--gt-json",
            str(task_b_json),
            "--pred-json",
            str(pred_task_b),
            "--subset",
            "all",
        ]
    )

    run(
        [
            py,
            str(SCRIPTS / "log_experiment.py"),
            "--experiment-id",
            "mock-task-a-perfect",
            "--task",
            "task_a",
            "--model",
            "mock_model",
            "--split",
            "all",
            "--metric",
            "top1_accuracy=1.0",
            "--metric",
            "macro_f1=1.0",
            "--notes",
            "auto-smoke",
        ]
    )
    run([py, str(SCRIPTS / "summarize_results.py"), "--task", "task_a"])

    print("[DONE] smoke_test_pipeline completed.")


if __name__ == "__main__":
    main()
