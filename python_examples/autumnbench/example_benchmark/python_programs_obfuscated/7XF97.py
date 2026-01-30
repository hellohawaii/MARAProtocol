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


def _positions_set(cells: List[Dict[str, Any]]) -> set[Tuple[int, int]]:
    positions: set[Tuple[int, int]] = set()
    for cell in cells:
        if isinstance(cell, dict):
            pos = _cell_pos(cell)
            if pos is not None:
                positions.add(pos)
    return positions


def _shift_cells(cells: List[Dict[str, Any]], dx: int, dy: int) -> List[Dict[str, Any]]:
    moved = copy.deepcopy(cells)
    for cell in moved:
        pos = cell.get("position")
        if isinstance(pos, dict):
            pos["x"] = _safe_int(pos.get("x")) + dx
            pos["y"] = _safe_int(pos.get("y")) + dy
    return moved


def _within_bounds(cell: Dict[str, Any], grid_size: int) -> bool:
    pos = _cell_pos(cell)
    if pos is None:
        return False
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size


def _intersects(cells_a: List[Dict[str, Any]], cells_b: List[Dict[str, Any]]) -> bool:
    if not cells_a or not cells_b:
        return False
    return bool(_positions_set(cells_a) & _positions_set(cells_b))


def _add_obj(prev_list: List[Dict[str, Any]], obj: Any) -> List[Dict[str, Any]]:
    combined = copy.deepcopy(prev_list)
    if isinstance(obj, list):
        combined.extend(copy.deepcopy(obj))
    elif isinstance(obj, dict):
        combined.append(copy.deepcopy(obj))
    return combined


def _water_cell(x: int, y: int) -> Dict[str, Any]:
    return {"position": {"x": x, "y": y}, "color": "blue"}


def _leaf_cell(color: str, x: int, y: int) -> Dict[str, Any]:
    return {"position": {"x": x, "y": y}, "color": color}


def _sun_cells(origin: Tuple[int, int]) -> List[Dict[str, Any]]:
    ox, oy = origin
    cells = []
    for dy in range(3):
        for dx in range(3):
            cells.append({"position": {"x": ox + dx, "y": oy + dy}, "color": "gold"})
    return cells


def _cloud_cells(origin: Tuple[int, int]) -> List[Dict[str, Any]]:
    ox, oy = origin
    rels = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
    ]
    return [{"position": {"x": ox + dx, "y": oy + dy}, "color": "gray"} for dx, dy in rels]


def _infer_sun_origin(cells: List[Dict[str, Any]]) -> Tuple[int, int]:
    positions = [_cell_pos(cell) for cell in cells]
    positions = [pos for pos in positions if pos is not None]
    if not positions:
        return 0, 0
    min_x = min(pos[0] for pos in positions)
    min_y = min(pos[1] for pos in positions)
    return min_x, min_y


def _infer_cloud_origin(cells: List[Dict[str, Any]]) -> Tuple[int, int]:
    positions = [_cell_pos(cell) for cell in cells]
    positions = [pos for pos in positions if pos is not None]
    if not positions:
        return 0, 0
    min_x = min(pos[0] for pos in positions)
    min_y = min(pos[1] for pos in positions)
    return min_x + 1, min_y


def init_state():
    grid_size = 16
    sun_origin = (0, 0)
    cloud_origin = (13, 0)

    leaves = [
        _leaf_cell("green", 1, 15),
        _leaf_cell("green", 3, 15),
        _leaf_cell("green", 5, 15),
        _leaf_cell("green", 7, 15),
        _leaf_cell("green", 9, 15),
        _leaf_cell("green", 11, 15),
        _leaf_cell("green", 13, 15),
        _leaf_cell("green", 15, 15),
    ]

    state = {
        "GRID_SIZE": grid_size,
        "1TH": _sun_cells(sun_origin),
        "gUe": _cloud_cells(cloud_origin),
        "LrC": [],
        "ICf": leaves,
    }

    hidden_state = {
        "sun_origin": sun_origin,
        "cloud_origin": cloud_origin,
        "sun_movingLeft": False,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 16), 16)
    prev_water = copy.deepcopy(state.get("LrC", []))
    prev_leaves = copy.deepcopy(state.get("ICf", []))
    prev_sun_cells = copy.deepcopy(state.get("1TH", []))
    prev_cloud_cells = copy.deepcopy(state.get("gUe", []))

    if hidden_state is None:
        hidden_state = {}

    prev_sun_origin = hidden_state.get("sun_origin")
    if not isinstance(prev_sun_origin, tuple):
        prev_sun_origin = _infer_sun_origin(prev_sun_cells)

    prev_cloud_origin = hidden_state.get("cloud_origin")
    if not isinstance(prev_cloud_origin, tuple):
        prev_cloud_origin = _infer_cloud_origin(prev_cloud_cells)

    prev_sun_moving_left = bool(hidden_state.get("sun_movingLeft", False))

    current_water = copy.deepcopy(prev_water)
    current_leaves = copy.deepcopy(prev_leaves)
    current_sun_origin = prev_sun_origin
    current_cloud_origin = prev_cloud_origin
    current_sun_moving_left = prev_sun_moving_left
    current_sun_cells = copy.deepcopy(prev_sun_cells)
    current_cloud_cells = copy.deepcopy(prev_cloud_cells)

    updated_vars: set[str] = set()
    action_type = _action_type(action)

    # --- on down ---
    if action_type == "down":
        drop_x = prev_cloud_origin[0]
        drop_y = prev_cloud_origin[1] + 1
        current_water = _add_obj(prev_water, _water_cell(drop_x, drop_y))
        updated_vars.add("LrC")

    # --- on water hits leaves ---
    moved_prev_water = _shift_cells(prev_water, 0, 1)
    if _intersects(moved_prev_water, prev_leaves):
        filtered = []
        for obj in current_water:
            if not _intersects(_shift_cells([obj], 0, 1), prev_leaves):
                filtered.append(copy.deepcopy(obj))
        current_water = filtered
        updated_vars.add("LrC")

    # --- on growth ---
    green_prev_leaves = [leaf for leaf in prev_leaves if leaf.get("color") == "green"]
    if _intersects(moved_prev_water, green_prev_leaves) and not _intersects(prev_sun_cells, prev_cloud_cells):
        grown = []
        for leaf in prev_leaves:
            if _intersects(_shift_cells([leaf], 0, -1), prev_water):
                pos = _cell_pos(leaf)
                if pos is None:
                    continue
                new_x, new_y = pos[0], pos[1] - 1
                new_color = "mediumpurple" if new_y == 12 else "green"
                grown.append(_leaf_cell(new_color, new_x, new_y))
        current_leaves = _add_obj(prev_leaves, grown)
        updated_vars.add("ICf")

    # --- on left/right (cloud) ---
    if action_type == "left":
        current_cloud_origin = (prev_cloud_origin[0] - 1, prev_cloud_origin[1])
        current_cloud_cells = _cloud_cells(current_cloud_origin)
        updated_vars.add("gUe")
    elif action_type == "right":
        current_cloud_origin = (prev_cloud_origin[0] + 1, prev_cloud_origin[1])
        current_cloud_cells = _cloud_cells(current_cloud_origin)
        updated_vars.add("gUe")

    # --- sun boundary updates ---
    if prev_sun_origin[0] == 0:
        current_sun_moving_left = False
        updated_vars.add("1TH")
    if prev_sun_origin[0] == grid_size - 3:
        current_sun_moving_left = True
        updated_vars.add("1TH")

    # --- on clicked sun ---
    if action_type == "click" and isinstance(action, dict):
        click_pos = (_safe_int(action.get("x")), _safe_int(action.get("y")))
        if click_pos in _positions_set(prev_sun_cells):
            dx = -1 if prev_sun_moving_left else 1
            current_sun_origin = (prev_sun_origin[0] + dx, prev_sun_origin[1])
            current_sun_cells = _sun_cells(current_sun_origin)
            current_sun_moving_left = prev_sun_moving_left
            updated_vars.add("1TH")

    # --- next updates (initnext) ---
    if "LrC" not in updated_vars:
        moved = _shift_cells(prev_water, 0, 1)
        current_water = [cell for cell in moved if _within_bounds(cell, grid_size)]

    if "ICf" not in updated_vars:
        current_leaves = copy.deepcopy(prev_leaves)

    if "gUe" not in updated_vars:
        current_cloud_origin = prev_cloud_origin
        current_cloud_cells = copy.deepcopy(prev_cloud_cells)

    if "1TH" not in updated_vars:
        current_sun_origin = prev_sun_origin
        current_sun_cells = copy.deepcopy(prev_sun_cells)
        current_sun_moving_left = prev_sun_moving_left

    new_state = {
        "GRID_SIZE": grid_size,
        "1TH": current_sun_cells,
        "gUe": current_cloud_cells,
        "LrC": current_water,
        "ICf": current_leaves,
    }

    next_hidden = {
        "sun_origin": current_sun_origin,
        "cloud_origin": current_cloud_origin,
        "sun_movingLeft": bool(current_sun_moving_left),
    }
    return new_state, next_hidden
