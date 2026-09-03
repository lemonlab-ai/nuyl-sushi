import argparse
from pathlib import Path


CHOICES_OLD = "'HMDB51', 'Diving48', 'Kinetics-710', 'MIT'"
CHOICES_NEW = "'HMDB51', 'Diving48', 'Kinetics-710', 'MIT', 'NTNU_SUSHI'"

NTNU_BRANCH = """
    elif args.data_set == 'NTNU_SUSHI':
        if not args.sparse_sample:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=args.num_frames,
                frame_sample_rate=args.sampling_rate,
                num_segment=1,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=False,
                args=args)
        else:
            dataset = VideoClsDataset(
                anno_path=anno_path,
                data_root=args.data_root,
                mode=mode,
                clip_len=1,
                frame_sample_rate=1,
                num_segment=args.num_frames,
                test_num_segment=args.test_num_segment,
                test_num_crop=args.test_num_crop,
                num_crop=1 if not test_mode else 3,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                short_side_size=args.short_side_size,
                new_height=256,
                new_width=320,
                sparse_sample=True,
                args=args)
        nb_classes = args.nb_classes
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch VideoMAEv2 for NTNU_SUSHI custom class count")
    parser.add_argument("--repo", required=True, help="Path to external/VideoMAEv2")
    return parser.parse_args()


def patch_run_class_finetuning(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "'NTNU_SUSHI'" in text:
        return False
    if CHOICES_OLD not in text:
        raise RuntimeError(f"Could not find expected dataset choices block in {path}")
    text = text.replace(CHOICES_OLD, CHOICES_NEW)
    path.write_text(text, encoding="utf-8")
    return True


def patch_build(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "elif args.data_set == 'NTNU_SUSHI':" in text:
        return False
    anchor = "    else:\n        raise NotImplementedError('Unsupported Dataset')"
    if anchor not in text:
        raise RuntimeError(f"Could not find insertion anchor in {path}")
    text = text.replace(anchor, NTNU_BRANCH + "\n" + anchor)
    path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        raise FileNotFoundError(repo)

    run_cls = repo / "run_class_finetuning.py"
    build_py = repo / "dataset" / "build.py"
    if not run_cls.exists():
        raise FileNotFoundError(run_cls)
    if not build_py.exists():
        raise FileNotFoundError(build_py)

    changed_run = patch_run_class_finetuning(run_cls)
    changed_build = patch_build(build_py)

    if changed_run or changed_build:
        print("[DONE] VideoMAEv2 patched for NTNU_SUSHI.")
    else:
        print("[SKIP] VideoMAEv2 already patched.")


if __name__ == "__main__":
    main()
