"""Utilities for resolving base and generated variant environments."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from planning_utils import load_color_dict, load_planning_data


_VARIANT_ENV_PATTERN = re.compile(r"^(?P<base>[A-Za-z0-9]+)_v\d+$")
# Subtask variants: e.g. 7XF97_v0002_move_cloud_t01 -> dir 7XF97-move_cloud
_SUBTASK_VARIANT_PATTERN = re.compile(
    r"^(?P<base>[A-Za-z0-9]+)_v\d+_(?P<phase>[a-z_]+)_t\d+$"
)


def get_variant_base_env_name(env_name: str) -> Optional[str]:
    match = _VARIANT_ENV_PATTERN.match(env_name)
    if not match:
        return None
    return match.group("base")


def _get_subtask_dir_name(env_name: str) -> Optional[str]:
    """Return the directory name for a subtask variant, e.g. '7XF97-move_cloud'."""
    match = _SUBTASK_VARIANT_PATTERN.match(env_name)
    if not match:
        return None
    return f"{match.group('base')}-{match.group('phase')}"


def is_variant_env_name(env_name: str) -> bool:
    return get_variant_base_env_name(env_name) is not None or _get_subtask_dir_name(env_name) is not None


def resolve_variant_env_dir(base_benchmark_dir: Path, env_name: str) -> Optional[Path]:
    # Check subtask pattern first (more specific)
    subtask_dir = _get_subtask_dir_name(env_name)
    if subtask_dir:
        env_dir = (base_benchmark_dir / "program_with_testcases" / subtask_dir).resolve()
        return env_dir if env_dir.is_dir() else None
    # Fall back to base variant pattern
    base_name = get_variant_base_env_name(env_name)
    if not base_name:
        return None
    env_dir = (base_benchmark_dir / "program_with_testcases" / base_name).resolve()
    return env_dir if env_dir.is_dir() else None


def resolve_program_path(base_benchmark_dir: Path, env_name: str) -> Tuple[Path, Path]:
    variant_dir = resolve_variant_env_dir(base_benchmark_dir, env_name)
    if variant_dir is not None:
        return variant_dir, (variant_dir / "programs" / f"{env_name}.sexp").resolve()
    return base_benchmark_dir.resolve(), (base_benchmark_dir / "programs" / f"{env_name}.sexp").resolve()


def load_program_text(base_benchmark_dir: Path, env_name: str) -> Tuple[Path, str]:
    data_dir, program_path = resolve_program_path(base_benchmark_dir, env_name)
    if not program_path.is_file():
        raise FileNotFoundError(f"Program not found: {program_path}")
    return data_dir, program_path.read_text(encoding="utf-8")


def load_goal_and_mask(base_benchmark_dir: Path, env_name: str):
    data_dir = resolve_variant_env_dir(base_benchmark_dir, env_name) or base_benchmark_dir.resolve()
    return data_dir, load_planning_data(data_dir, env_name)


def load_color_dict_for_env(base_benchmark_dir: Path, env_name: str) -> Dict[int, str]:
    data_dir = resolve_variant_env_dir(base_benchmark_dir, env_name) or base_benchmark_dir.resolve()
    return load_color_dict(data_dir)
