"""Shared utilities for loading and converting planning goal data."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_color_dict(data_dir: Path) -> Dict[int, str]:
    path = data_dir / "color_dict.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"color_dict.yaml not found at {path}")
    result: Dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split(":", 1)
        if len(parts) == 2:
            result[int(parts[0].strip())] = parts[1].strip()
    return result


def load_planning_data(data_dir: Path, env_name: str) -> Optional[Tuple[List[List[int]], List[List[int]]]]:
    """Load raw goal and mask integer matrices from the planning JSON.

    Returns (goal, mask) or None if the file doesn't exist.
    """
    planning_path = data_dir / "prompts" / f"{env_name}_planning.json"
    if not planning_path.is_file():
        return None
    planning_data = json.loads(planning_path.read_text(encoding="utf-8"))
    return planning_data["goal"], planning_data["mask"]


def goal_to_color_grid(raw_goal: List[List[int]], color_dict: Dict[int, str]) -> List[List[str]]:
    return [
        [color_dict.get(cell, str(cell)) for cell in row]
        for row in raw_goal
    ]


def color_grid_to_scene_graph(goal_color: List[List[str]]) -> Dict[str, Any]:
    grid_size = len(goal_color)
    sg: Dict[str, Any] = {"GRID_SIZE": grid_size}
    for r in range(grid_size):
        for c in range(grid_size):
            val = goal_color[r][c]
            if val not in sg:
                sg[val] = []
            sg[val].append({"position": {"x": c, "y": r}})
    return sg


def mask_to_positions(raw_mask: List[List[int]]) -> List[Dict[str, int]]:
    grid_size = len(raw_mask)
    return [
        {"x": c, "y": r}
        for r in range(grid_size)
        for c in range(grid_size)
        if raw_mask[r][c] == 1
    ]


def scene_graph_to_color_grid(scene_graph: Dict[str, Any], background_color: str) -> List[List[str]]:
    grid_size = scene_graph.get("GRID_SIZE", 0)
    matrix = [[background_color] * grid_size for _ in range(grid_size)]
    for key, objects in scene_graph.items():
        if key == "GRID_SIZE":
            continue
        for obj in objects:
            x = obj["position"]["x"]
            y = obj["position"]["y"]
            color = obj.get("color", key).lower()
            if 0 <= y < grid_size and 0 <= x < grid_size:
                matrix[y][x] = color
    return matrix


def check_grid_same(grid1: List[List[str]], grid2: List[List[str]], mask: List[List[int]]) -> bool:
    for i in range(len(grid1)):
        for j in range(len(grid1[0])):
            if not mask[i][j]:
                continue
            if grid1[i][j] != grid2[i][j]:
                return False
    return True
