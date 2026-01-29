import copy
from typing import Any, Dict, List, Tuple


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _action_type(action: Any) -> str:
    if isinstance(action, dict):
        return str(action.get("type", "noop"))
    return str(action) if action is not None else "noop"


def _cell_pos(cell: Dict[str, Any]) -> Tuple[int, int] | None:
    pos = cell.get("position")
    if isinstance(pos, dict) and "x" in pos and "y" in pos:
        return _safe_int(pos["x"]), _safe_int(pos["y"])
    return None


def _move_cells(cells: List[Dict[str, Any]], dx: int, dy: int) -> List[Dict[str, Any]]:
    moved = copy.deepcopy(cells)
    for cell in moved:
        pos = cell.get("position")
        if isinstance(pos, dict):
            pos["x"] = _safe_int(pos.get("x")) + dx
            pos["y"] = _safe_int(pos.get("y")) + dy
    return moved


def _within_bounds(pos: Tuple[int, int], grid_size: int) -> bool:
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size


def _filter_within_bounds(cells: List[Dict[str, Any]], grid_size: int) -> List[Dict[str, Any]]:
    filtered = []
    for cell in cells:
        pos = _cell_pos(cell)
        if pos is not None and _within_bounds(pos, grid_size):
            filtered.append(copy.deepcopy(cell))
    return filtered


def _cloud_cells(grid_size: int) -> List[Dict[str, Any]]:
    cells = []
    for y in range(2):
        for x in range(grid_size):
            cells.append({"position": {"x": x, "y": y}, "color": "gray"})
    return cells


def _spawn_water_cells() -> List[Dict[str, Any]]:
    positions = [(2, 2), (6, 2), (10, 2), (14, 2)]
    return [{"position": {"x": x, "y": y}, "color": "lightblue"} for x, y in positions]


def init_state():
    grid_size = 17
    state = {
        "GRID_SIZE": grid_size,
        "cloud": _cloud_cells(grid_size),
        "water": [],
    }
    hidden_state = {
        "wind": 0,
        "time": 0,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 17), 17)
    prev_water = copy.deepcopy(state.get("water", []))

    if hidden_state is None:
        _, hidden_state = init_state()

    prev_wind = int(hidden_state.get("wind", 0))
    prev_time = int(hidden_state.get("time", 0))

    current_water = copy.deepcopy(prev_water)
    current_wind = prev_wind
    updated_vars: set[str] = set()

    # --- on wind movement (uses prev water) ---
    if prev_wind == 0:
        current_water = _move_cells(prev_water, 0, 1)
        updated_vars.add("water")
    elif prev_wind == 1:
        current_water = _move_cells(prev_water, 1, 1)
        updated_vars.add("water")
    elif prev_wind == -1:
        current_water = _move_cells(prev_water, -1, 1)
        updated_vars.add("water")

    # --- on left/right (wind control) ---
    action_type = _action_type(action)
    if action_type == "left":
        current_wind = prev_wind if prev_wind == -1 else prev_wind - 1
        updated_vars.add("wind")
    elif action_type == "right":
        current_wind = prev_wind if prev_wind == 1 else prev_wind + 1
        updated_vars.add("wind")

    # --- on time-based spawn ---
    if prev_time % 5 == 2:
        current_water = copy.deepcopy(current_water)
        current_water.extend(_spawn_water_cells())
        updated_vars.add("water")

    # --- next updates ---
    if "water" not in updated_vars:
        current_water = _filter_within_bounds(prev_water, grid_size)

    if "wind" not in updated_vars:
        current_wind = prev_wind

    current_time = prev_time + 1

    new_state = {
        "GRID_SIZE": grid_size,
        "cloud": _cloud_cells(grid_size),
        "water": current_water,
    }
    next_hidden = {
        "wind": int(current_wind),
        "time": int(current_time),
    }
    return new_state, next_hidden
