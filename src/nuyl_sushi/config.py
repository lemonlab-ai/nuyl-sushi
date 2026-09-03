import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT_ENV = "NUYL_SUSHI_ROOT"


def _parents_including_self(path: Path) -> Iterable[Path]:
    yield path
    yield from path.parents


def find_project_root(start: Optional[Path] = None) -> Path:
    """Resolve the checkout root without embedding a machine-specific path."""
    override = os.environ.get(PROJECT_ROOT_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve()])
    for candidate in candidates:
        base = candidate if candidate.is_dir() else candidate.parent
        for parent in _parents_including_self(base):
            if (parent / "pyproject.toml").is_file() and (parent / "src" / "nuyl_sushi").is_dir():
                return parent
    return Path.cwd().resolve()


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Optional[Path] = None) -> "ProjectPaths":
        return cls(root=find_project_root(start))

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

