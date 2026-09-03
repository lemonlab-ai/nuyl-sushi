from typing import Any, Dict, List, Optional


def parse_wandb_tags(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def init_wandb_run(
    enabled: bool,
    project: str,
    entity: str = "",
    run_name: str = "",
    group: str = "",
    tags: Optional[List[str]] = None,
    mode: str = "online",
    config: Optional[Dict[str, Any]] = None,
    job_type: str = "",
) -> Any:
    if not enabled:
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "WandB is enabled but package `wandb` is not installed. "
            "Please install it in the active environment."
        ) from exc

    kwargs: Dict[str, Any] = {
        "project": project,
        "mode": mode,
        "config": config or {},
        "reinit": "finish_previous",
    }
    if entity.strip():
        kwargs["entity"] = entity.strip()
    if run_name.strip():
        kwargs["name"] = run_name.strip()
    if group.strip():
        kwargs["group"] = group.strip()
    if job_type.strip():
        kwargs["job_type"] = job_type.strip()
    if tags:
        kwargs["tags"] = list(tags)
    return wandb.init(**kwargs)


def wandb_log(run: Any, payload: Dict[str, Any]) -> None:
    if run is None:
        return
    run.log(payload)


def finish_wandb_run(run: Any) -> None:
    if run is None:
        return
    run.finish()
