import csv
import json
import subprocess
import sys
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]


def run(cmd: List[str], cwd: Path = ROOT_DIR) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_tiny_master(path: Path) -> None:
    videos = {
        "vid_s1_01": {
            "source_path": "videos/vid_s1_01.mp4",
            "subject_id": "S1",
            "annotations": [{"coarse_label": "PREPARE", "fine_label": "WASH"}],
        },
        "vid_s1_02": {
            "source_path": "videos/vid_s1_02.mp4",
            "subject_id": "S1",
            "annotations": [{"coarse_label": "CUTTING", "fine_label": "SLICE"}],
        },
        "vid_s2_01": {
            "source_path": "videos/vid_s2_01.mp4",
            "subject_id": "S2",
            "annotations": [{"coarse_label": "PREPARE", "fine_label": "WASH"}],
        },
        "vid_s3_01": {
            "source_path": "videos/vid_s3_01.mp4",
            "subject_id": "S3",
            "annotations": [{"coarse_label": "CUTTING", "fine_label": "SLICE"}],
        },
    }
    write_json(path, {"meta": {"source": "ci_smoke"}, "videos": videos})


def prepare_task_a_inputs(split_root: Path) -> None:
    class_map = {"WASH": 0, "SLICE": 1}
    for fold_dir in sorted([p for p in split_root.glob("fold_*") if p.is_dir()]):
        master = load_json(fold_dir / "master_annotations.json")
        videos = master.get("videos", {})
        task_a_dir = fold_dir / "task_a"
        task_a_dir.mkdir(parents=True, exist_ok=True)
        write_json(task_a_dir / "class_map_fine.json", class_map)

        manifest_rows = []
        pred_rows = []
        for vid, rec in sorted(videos.items()):
            anns = rec.get("annotations", [])
            fine = ""
            if isinstance(anns, list) and anns and isinstance(anns[0], dict):
                fine = str(anns[0].get("fine_label", "")).strip()
            if fine not in class_map:
                continue
            label_id = int(class_map[fine])
            subset = str(rec.get("subset", "train"))
            clip_id = str(vid)
            manifest_rows.append(
                {
                    "clip_id": clip_id,
                    "label_id": label_id,
                    "subset": subset,
                }
            )
            pred_rows.append(
                {
                    "clip_id": clip_id,
                    "pred_label_id": label_id,
                }
            )

        with (task_a_dir / "task_a_fine_manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_id", "label_id", "subset"])
            writer.writeheader()
            writer.writerows(manifest_rows)

        with (task_a_dir / "raw_pred.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["clip_id", "pred_label_id"])
            writer.writeheader()
            writer.writerows(pred_rows)


def to_activitynet_subset(subset: str) -> str:
    s = str(subset).strip().lower()
    if s == "train":
        return "training"
    if s == "val":
        return "validation"
    if s == "test":
        return "testing"
    return "training"


def prepare_task_b_inputs(split_root: Path) -> None:
    for fold_dir in sorted([p for p in split_root.glob("fold_*") if p.is_dir()]):
        master = load_json(fold_dir / "master_annotations.json")
        videos = master.get("videos", {})
        task_b_dir = fold_dir / "task_b"
        task_b_dir.mkdir(parents=True, exist_ok=True)

        database = {}
        pred_rows = []
        for vid, rec in sorted(videos.items()):
            anns = rec.get("annotations", [])
            label = ""
            if isinstance(anns, list) and anns and isinstance(anns[0], dict):
                label = str(anns[0].get("fine_label", "")).strip()
            if not label:
                continue
            subset = to_activitynet_subset(str(rec.get("subset", "train")))
            database[str(vid)] = {
                "duration": 10.0,
                "subset": subset,
                "annotations": [{"label": label, "segment": [0.0, 1.0]}],
            }
            if subset == "testing":
                pred_rows.append(
                    {
                        "video_id": str(vid),
                        "label": label,
                        "start": 0.0,
                        "end": 1.0,
                        "score": 0.99,
                    }
                )

        write_json(task_b_dir / "activitynet_fine.json", {"database": database})
        with (task_b_dir / "raw_pred.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "label", "start", "end", "score"])
            writer.writeheader()
            writer.writerows(pred_rows)


def assert_resume_did_not_rewrite(task_a_metrics_path: Path, old_mtime_ns: int) -> None:
    new_mtime_ns = task_a_metrics_path.stat().st_mtime_ns
    if new_mtime_ns != old_mtime_ns:
        raise RuntimeError(
            f"Resume check failed: metrics file was rewritten ({old_mtime_ns} -> {new_mtime_ns})."
        )


def assert_handoff_ready(manifest_path: Path) -> None:
    payload = load_json(manifest_path)
    if not bool(payload.get("ready_for_outer_loop", False)):
        raise RuntimeError("Handoff manifest is not ready_for_outer_loop=true.")
    missing = payload.get("missing_required_artifacts", [])
    if isinstance(missing, list) and missing:
        raise RuntimeError(f"Handoff manifest still missing required artifacts: {missing}")


def prepare_skill_level_keypoint_index(root: Path) -> Path:
    seq_dir = root / "skill_level_sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    index_csv = root / "skill_level_index.csv"

    def mk_hand_2d(v: float) -> List[List[float]]:
        return [[v, v, 1.0] for _ in range(21)]

    def mk_hand_3d(v: float) -> List[List[float]]:
        return [[v, v, v, 1.0] for _ in range(21)]

    def write_seq(path: Path, base: float) -> None:
        frames = []
        for i in range(4):
            vv = base + 0.01 * i
            frames.append(
                {
                    "frame_idx": i,
                    "left": mk_hand_2d(vv),
                    "right": mk_hand_2d(vv + 0.001),
                    "left_3d": mk_hand_3d(vv),
                    "right_3d": mk_hand_3d(vv + 0.001),
                    "left_world_3d": mk_hand_3d(vv),
                    "right_world_3d": mk_hand_3d(vv + 0.001),
                }
            )
        write_json(path, {"fps": 30.0, "frames": frames})

    rows = [
        ("s1_beg", "S1", "beginner", 0.10),
        ("s1_int", "S1", "intermediate", 0.30),
        ("s2_beg", "S2", "beginner", 0.11),
        ("s2_exp", "S2", "expert", 0.60),
        ("s3_int", "S3", "intermediate", 0.31),
        ("s3_exp", "S3", "expert", 0.61),
    ]

    with index_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "subject_id", "skill_level", "sequence_path"],
        )
        writer.writeheader()
        for sample_id, subject_id, skill_level, base in rows:
            seq_path = seq_dir / f"{sample_id}.json"
            write_seq(seq_path, base=base)
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "subject_id": subject_id,
                    "skill_level": skill_level,
                    "sequence_path": str(seq_path),
                }
            )

    return index_csv


def run_optional_mediapipe_extract_smoke(root: Path) -> None:
    has_mediapipe = importlib.util.find_spec("mediapipe") is not None
    has_cv2 = importlib.util.find_spec("cv2") is not None
    if not has_mediapipe or not has_cv2:
        print("[SKIP] Optional mediapipe extract smoke (mediapipe/cv2 not available).")
        return

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        print("[SKIP] Optional mediapipe extract smoke (cannot import cv2/numpy).")
        return

    video_dir = root / "mp_extract" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / "smoke.mp4"

    width, height = 64, 64
    fps = 10.0
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print("[SKIP] Optional mediapipe extract smoke (cannot create test video).")
        return
    for i in range(12):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 17) % 255
        frame[:, :, 1] = (i * 9) % 255
        frame[:, :, 2] = (i * 3) % 255
        writer.write(frame)
    writer.release()

    master_json = root / "mp_extract" / "master_annotations.json"
    write_json(
        master_json,
        {
            "videos": {
                "smoke_video": {
                    "video_id": "smoke_video",
                    "source_path": str(video_path.resolve()),
                    "subset": "train",
                    "subject_id": "S_MP",
                    "skill_level": "beginner",
                    "annotations": [],
                }
            }
        },
    )

    out_dir = root / "mp_extract" / "out"
    run(
        [
            sys.executable,
            "scripts/extract_skill_level_keypoints.py",
            "--master-json",
            str(master_json),
            "--output-dir",
            str(out_dir),
            "--max-videos",
            "1",
            "--max-frames",
            "8",
        ]
    )
    index_csv = out_dir / "skill_level_keypoint_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(index_csv)
    rows = list(csv.DictReader(index_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise RuntimeError("extract_skill_level_keypoints smoke produced empty index.")
    status = str(rows[0].get("status", "")).strip().lower()
    if status != "ok":
        raise RuntimeError(f"extract_skill_level_keypoints smoke status is not ok: {status}")


def main() -> None:
    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "scripts/generate_subject_kfold_masters.py",
            "scripts/run_kfold_experiments.py",
            "scripts/run_seed_sweep.py",
            "scripts/export_paper_tables.py",
            "scripts/run_kfold_pipeline.py",
            "scripts/build_outer_loop_handoff.py",
            "scripts/export_skill_level_manifest.py",
            "scripts/extract_skill_level_keypoints.py",
            "scripts/eval_skill_level_from_keypoints.py",
            "scripts/audit_source_dataset.py",
            "scripts/convert_via_to_master.py",
            "scripts/build_metadata_candidates.py",
            "scripts/validate_master_dataset.py",
            "src/nuyl_sushi/__init__.py",
            "src/nuyl_sushi/_version.py",
            "src/nuyl_sushi/config.py",
            "src/nuyl_sushi/logging.py",
            "src/nuyl_sushi/data/metadata_candidates.py",
            "src/nuyl_sushi/data/master.py",
            "src/nuyl_sushi/data/via.py",
            "src/nuyl_sushi/data/conversion.py",
            "src/nuyl_sushi/domain/models.py",
            "src/nuyl_sushi/cli/metadata_candidates.py",
            "src/nuyl_sushi/cli/validate_master.py",
            "src/nuyl_sushi/cli/convert_via.py",
            "apps/streamlit_app.py",
            "apps/gesture_coach/coach.py",
            "apps/gesture_coach/mano_adapter.py",
        ]
    )

    for script in [
        "scripts/generate_subject_kfold_masters.py",
        "scripts/run_kfold_experiments.py",
        "scripts/run_seed_sweep.py",
        "scripts/export_paper_tables.py",
        "scripts/build_outer_loop_handoff.py",
        "scripts/export_skill_level_manifest.py",
        "scripts/extract_skill_level_keypoints.py",
        "scripts/eval_skill_level_from_keypoints.py",
        "scripts/audit_source_dataset.py",
        "scripts/convert_via_to_master.py",
        "scripts/build_metadata_candidates.py",
        "scripts/validate_master_dataset.py",
    ]:
        run([sys.executable, script, "--help"])

    run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
        ]
    )

    run(
        [
            sys.executable,
            "-c",
            (
                "from apps.gesture_coach.coach import compare_sequences; "
                "from apps.gesture_coach.mano_adapter import ensure_mano_sequence; "
                "seq={'frames':[{'left_world_3d':[[0,0,0,1]]*21,'right_world_3d':[[0,0,0,1]]*21}]}; "
                "seq_m=ensure_mano_sequence(seq,allow_world_fallback=True); "
                "out=compare_sequences(seq_m,seq_m,hand_mode='both',coord_mode='3d_mano',max_frames=10); "
                "assert float(out['score_0_100']) >= 99.0"
            ),
        ]
    )

    with tempfile.TemporaryDirectory(prefix="ntnu_sushi_ci_smoke_") as tmp:
        tmp_root = Path(tmp)
        master_json = tmp_root / "master_annotations.json"
        split_root = tmp_root / "splits"
        out_root = tmp_root / "kfold_out"
        sweep_root = tmp_root / "seed_sweep"
        handoff_root = tmp_root / "outer_handoff"
        build_tiny_master(master_json)

        run(
            [
                sys.executable,
                "scripts/generate_subject_kfold_masters.py",
                "--master-json",
                str(master_json),
                "--output-dir",
                str(split_root),
                "--num-folds",
                "2",
                "--val-ratio",
                "0.2",
                "--seed",
                "7",
                "--stratify-level",
                "fine",
                "--subject-leakage-policy",
                "fail",
                "--imbalance-policy",
                "warn",
                "--max-label-relative-deviation",
                "2.0",
                "--min-test-label-count-per-fold",
                "1",
                "--min-count-policy",
                "warn",
            ]
        )
        prepare_task_a_inputs(split_root)
        prepare_task_b_inputs(split_root)

        run(
            [
                sys.executable,
                "scripts/run_kfold_experiments.py",
                "--split-root",
                str(split_root),
                "--output-root",
                str(out_root),
                "--model-name",
                "ci_smoke_model",
                "--run-task-a",
                "--run-task-b",
                "--task-a-raw-pred-rel",
                "task_a/raw_pred.csv",
                "--task-a-raw-format",
                "csv",
                "--task-a-subset",
                "test",
                "--task-b-raw-pred-rel",
                "task_b/raw_pred.csv",
                "--task-b-raw-format",
                "csv",
                "--task-b-subset",
                "testing",
                "--export-paper-table",
            ]
        )
        fold0_metrics = out_root / "ci_smoke_model" / "fold_00" / "task_a_metrics.json"
        if not fold0_metrics.exists():
            raise FileNotFoundError(fold0_metrics)
        old_mtime_ns = fold0_metrics.stat().st_mtime_ns
        fold0_task_b_metrics = out_root / "ci_smoke_model" / "fold_00" / "task_b_metrics.json"
        if not fold0_task_b_metrics.exists():
            raise FileNotFoundError(fold0_task_b_metrics)
        old_task_b_mtime_ns = fold0_task_b_metrics.stat().st_mtime_ns

        run(
            [
                sys.executable,
                "scripts/run_kfold_experiments.py",
                "--split-root",
                str(split_root),
                "--output-root",
                str(out_root),
                "--model-name",
                "ci_smoke_model",
                "--run-task-a",
                "--run-task-b",
                "--task-a-raw-pred-rel",
                "task_a/raw_pred.csv",
                "--task-a-raw-format",
                "csv",
                "--task-a-subset",
                "test",
                "--task-b-raw-pred-rel",
                "task_b/raw_pred.csv",
                "--task-b-raw-format",
                "csv",
                "--task-b-subset",
                "testing",
                "--resume",
            ]
        )
        assert_resume_did_not_rewrite(fold0_metrics, old_mtime_ns)
        assert_resume_did_not_rewrite(fold0_task_b_metrics, old_task_b_mtime_ns)

        run(
            [
                sys.executable,
                "scripts/run_seed_sweep.py",
                "--seeds",
                "11,12",
                "--model-name",
                "ci_smoke_model",
                "--output-root",
                str(sweep_root),
                "--run-task-a",
                "--run-task-b",
                "--task-a-raw-pred-rel",
                "task_a/raw_pred.csv",
                "--task-a-raw-format",
                "csv",
                "--task-a-subset",
                "test",
                "--task-b-raw-pred-rel",
                "task_b/raw_pred.csv",
                "--task-b-raw-format",
                "csv",
                "--task-b-subset",
                "testing",
                "--split-root",
                str(split_root),
                "--export-paper-table",
            ]
        )
        sweep_summary = sweep_root / "ci_smoke_model" / "seed_sweep_summary.json"
        if not sweep_summary.exists():
            raise FileNotFoundError(sweep_summary)

        run(
            [
                sys.executable,
                "scripts/export_paper_tables.py",
                "--experiment-summary",
                str(out_root / "ci_smoke_model" / "experiment_summary.json"),
                "--seed-sweep-summary",
                str(sweep_summary),
                "--no-ablation",
            ]
        )

        run(
            [
                sys.executable,
                "scripts/build_outer_loop_handoff.py",
                "--model-name",
                "ci_smoke_model",
                "--kfold-root",
                str(out_root),
                "--seed-sweep-root",
                str(sweep_root),
                "--handoff-root",
                str(handoff_root),
                "--strict",
            ]
        )
        manifest = handoff_root / "ci_smoke_model" / "handoff_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        assert_handoff_ready(manifest)

        skill_index = prepare_skill_level_keypoint_index(tmp_root)
        skill_out = tmp_root / "skill_level_out"
        run(
            [
                sys.executable,
                "scripts/eval_skill_level_from_keypoints.py",
                "--index-path",
                str(skill_index),
                "--coord-modes",
                "2d,3d_world,3d_mano",
                "--num-folds",
                "3",
                "--seed",
                "13",
                "--allow-mano-proxy",
                "--output-dir",
                str(skill_out),
            ]
        )
        if not (skill_out / "skill_level_keypoint_benchmark.json").exists():
            raise FileNotFoundError(skill_out / "skill_level_keypoint_benchmark.json")
        if not (skill_out / "skill_level_keypoint_benchmark.csv").exists():
            raise FileNotFoundError(skill_out / "skill_level_keypoint_benchmark.csv")

        run_optional_mediapipe_extract_smoke(tmp_root)

    print("[DONE] CI smoke checks passed.")


if __name__ == "__main__":
    main()
