import logging
from typing import Union


DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging(level: Union[int, str] = "INFO") -> None:
    """Configure predictable application logging at process entry points."""
    resolved = level
    if isinstance(level, str):
        resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=resolved, format=DEFAULT_FORMAT)

