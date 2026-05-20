"""Load configuration from .gitsight.toml or ~/.gitsight.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

_CONFIG_FILENAME = ".gitsight.toml"

_VALID_KEYS = {
    "model",
    "threshold",
    "min_cluster_size",
    "concurrency",
    "max_commits",
    "no_ai",
    "branch",
    "export_format",
    "timeline",
    "author_stats",
    "verbose",
}


def _load_raw(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML in {path}: {e}") from e


def load_config() -> dict[str, Any]:
    """Return merged config; local .gitsight.toml takes precedence over ~/.gitsight.toml."""
    home_cfg = _load_raw(Path.home() / _CONFIG_FILENAME)
    local_cfg = _load_raw(Path.cwd() / _CONFIG_FILENAME)
    merged: dict[str, Any] = {**home_cfg, **local_cfg}

    unknown = set(merged) - _VALID_KEYS
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(sorted(unknown))}")

    if "threshold" in merged:
        v = merged["threshold"]
        if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
            raise ValueError("Config 'threshold' must be a float between 0.0 and 1.0")
    if "min_cluster_size" in merged and not isinstance(merged["min_cluster_size"], int):
        raise ValueError("Config 'min_cluster_size' must be an integer")
    if "concurrency" in merged and not isinstance(merged["concurrency"], int):
        raise ValueError("Config 'concurrency' must be an integer")
    if "max_commits" in merged and not isinstance(merged["max_commits"], int):
        raise ValueError("Config 'max_commits' must be an integer")
    if "no_ai" in merged and not isinstance(merged["no_ai"], bool):
        raise ValueError("Config 'no_ai' must be a boolean")
    if "timeline" in merged and not isinstance(merged["timeline"], bool):
        raise ValueError("Config 'timeline' must be a boolean")
    if "author_stats" in merged and not isinstance(merged["author_stats"], bool):
        raise ValueError("Config 'author_stats' must be a boolean")
    if "export_format" in merged and merged["export_format"] not in {"json", "markdown"}:
        raise ValueError("Config 'export_format' must be 'json' or 'markdown'")

    return merged
