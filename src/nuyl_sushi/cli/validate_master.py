import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from nuyl_sushi.config import ProjectPaths
from nuyl_sushi.data.master import load_master, validate_master, validation_summary


def build_parser() -> argparse.ArgumentParser:
    default_master = ProjectPaths.discover().artifacts / "master" / "master_annotations.json"
    parser = argparse.ArgumentParser(description="Validate canonical master_annotations.json")
    parser.add_argument("--master-json", default=str(default_master))
    parser.add_argument("--no-check-files", action="store_true")
    parser.add_argument("--json-report", default="", help="Optional machine-readable report path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    dataset = load_master(Path(args.master_json))
    issues = validate_master(dataset, check_source_files=not args.no_check_files)
    summary = validation_summary(dataset, issues)

    print(f"Videos: {summary['videos']}")
    print(f"Annotations: {summary['annotations']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Errors: {summary['errors']}")
    for issue in issues[:30]:
        prefix = "ERR" if issue.severity == "error" else "WARN"
        print(f"{prefix}: [{issue.location}] {issue.message} ({issue.code})")
    if len(issues) > 30:
        print(f"... ({len(issues) - 30} more issues)")

    if args.json_report:
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
