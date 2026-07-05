"""Path helpers for commands that run from the repository root."""

from pathlib import Path

from churn_ml.config import PROJECT_ROOT


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()
