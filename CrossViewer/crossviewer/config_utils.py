"""Helpers for loading configs and resolving path fields."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import yaml


_PATH_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("model", "vision_encoder_path"),
    ("data", "data_root"),
    ("data", "jsonl_train"),
    ("data", "jsonl_val"),
    ("training", "deepspeed_config"),
    ("training", "save_dir"),
    ("training", "log_dir"),
    ("training", "resume_from"),
)


def validate_required_paths(config: Dict[str, Any], fields: Iterable[Tuple[str, str]]) -> None:
    """Raise a clear error when required config path fields are empty."""
    missing = []
    for section, key in fields:
        section_obj = config.get(section)
        value = section_obj.get(key) if isinstance(section_obj, dict) else None
        if value is None:
            missing.append(f"{section}.{key}")
            continue
        if isinstance(value, str) and value.strip() == "":
            missing.append(f"{section}.{key}")

    if missing:
        raise ValueError(
            "Missing required config path fields. Fill these values in your YAML before running: "
            + ", ".join(missing)
        )


def _resolve_path_value(raw_value: Any, base_dir: Path) -> Any:
    if raw_value is None or not isinstance(raw_value, str) or raw_value == "":
        return raw_value

    expanded = Path(raw_value).expanduser()
    if expanded.is_absolute():
        return str(expanded)

    candidate = base_dir / expanded
    if raw_value.startswith(("./", "../")) or candidate.exists():
        return str(candidate.resolve())

    return raw_value


def resolve_config_paths(config: Dict[str, Any], config_path: str | Path) -> Dict[str, Any]:
    """Resolve selected config paths relative to the config file."""
    resolved = deepcopy(config)
    base_dir = Path(config_path).expanduser().resolve().parent

    for section, key in _PATH_FIELDS:
        section_obj = resolved.get(section)
        if not isinstance(section_obj, dict) or key not in section_obj:
            continue
        section_obj[key] = _resolve_path_value(section_obj.get(key), base_dir)

    return resolved


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config and resolve supported path fields."""
    config_file = Path(config_path).expanduser().resolve()
    with config_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_file}")

    return resolve_config_paths(config, config_file)
