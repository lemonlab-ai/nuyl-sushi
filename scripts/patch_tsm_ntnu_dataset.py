import argparse
from pathlib import Path


NTNU_FUNCTION = """
def return_ntnu_sushi(modality):
    if modality != 'RGB':
        raise NotImplementedError('no such modality:' + modality)
    # Use env vars so this works without hard-coding machine-specific paths.
    root_data = os.getenv('NTNU_SUSHI_TSM_ROOT', 'ntnu_sushi/frames')
    filename_imglist_train = os.getenv('NTNU_SUSHI_TSM_TRAIN_LIST', 'ntnu_sushi/train_videofolder.txt')
    filename_imglist_val = os.getenv('NTNU_SUSHI_TSM_VAL_LIST', 'ntnu_sushi/val_videofolder.txt')
    filename_categories = os.getenv('NTNU_SUSHI_TSM_CATEGORIES', 'ntnu_sushi/category.txt')
    prefix = 'img_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
"""

JOIN_HELPER = """
def _join_root(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(ROOT_DATASET, path_value)
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch TSM dataset_config.py for ntnu_sushi")
    parser.add_argument("--repo", required=True, help="Path to external/temporal-shift-module")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo)
    if not repo.exists():
        raise FileNotFoundError(repo)
    path = repo / "ops" / "dataset_config.py"
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    changed = False

    if "def return_ntnu_sushi(modality):" not in text:
        anchor = "\ndef return_dataset(dataset, modality):\n"
        if anchor not in text:
            raise RuntimeError("Could not find return_dataset anchor.")
        text = text.replace(anchor, "\n" + NTNU_FUNCTION.strip() + "\n\n" + anchor.strip() + "\n")
        changed = True

    if "'ntnu_sushi': return_ntnu_sushi" not in text:
        target = "'kinetics': return_kinetics }"
        if target not in text:
            raise RuntimeError("Could not find dict_single target for insertion.")
        text = text.replace(target, "'kinetics': return_kinetics, 'ntnu_sushi': return_ntnu_sushi }")
        changed = True

    if "def _join_root(path_value):" not in text:
        insert_after = "ROOT_DATASET = '/ssd/video/'  # '/data/jilin/'\n"
        if insert_after not in text:
            raise RuntimeError("Could not find ROOT_DATASET declaration.")
        text = text.replace(insert_after, insert_after + "\n" + JOIN_HELPER.strip() + "\n")
        changed = True

    old_join = (
        "    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)\n"
        "    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)\n"
        "    if isinstance(file_categories, str):\n"
        "        file_categories = os.path.join(ROOT_DATASET, file_categories)\n"
    )
    new_join = (
        "    file_imglist_train = _join_root(file_imglist_train)\n"
        "    file_imglist_val = _join_root(file_imglist_val)\n"
        "    if isinstance(file_categories, str):\n"
        "        file_categories = _join_root(file_categories)\n"
    )
    if old_join in text:
        text = text.replace(old_join, new_join)
        changed = True

    path.write_text(text, encoding="utf-8")
    if changed:
        print("[DONE] TSM patched for ntnu_sushi.")
    else:
        print("[SKIP] TSM already patched.")


if __name__ == "__main__":
    main()
