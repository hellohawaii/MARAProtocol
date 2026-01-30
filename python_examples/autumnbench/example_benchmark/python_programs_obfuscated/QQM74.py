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


def _make_cell(x: int, y: int, color: str) -> Dict[str, Any]:
    return {"position": {"x": x, "y": y}, "color": color}


def _is_free_pos(pos: Tuple[int, int], cells: List[Dict[str, Any]]) -> bool:
    return pos not in _positions_set(cells)


def _move_cells(cells: List[Dict[str, Any]], dx: int, dy: int) -> List[Dict[str, Any]]:
    moved = copy.deepcopy(cells)
    for cell in moved:
        pos = cell.get("position")
        if isinstance(pos, dict):
            pos["x"] = _safe_int(pos.get("x")) + dx
            pos["y"] = _safe_int(pos.get("y")) + dy
    return moved


def init_state():
    grid_size = 21
    center = grid_size // 2
    blobs = [_make_cell(center, center, "blue")]
    state = {
        "GRID_SIZE": grid_size,
        "9Ew": blobs,
    }
    hidden_state = {
        "xVel": 0,
        "yVel": 0,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    if hidden_state is None:
        hidden_state = {}

    prev_blobs = copy.deepcopy(state.get("9Ew", []))
    prev_x_vel = int(hidden_state.get("xVel", 0))
    prev_y_vel = int(hidden_state.get("yVel", 0))

    current_blobs = copy.deepcopy(prev_blobs)
    action_type = _action_type(action)

    # on clicked: add blob at click if free (based on prev blobs)
    if action_type == "click" and isinstance(action, dict):
        click_pos = (_safe_int(action.get("x")), _safe_int(action.get("y")))
        if _is_free_pos(click_pos, prev_blobs):
            current_blobs.append(_make_cell(click_pos[0], click_pos[1], "blue"))

    # on left/right/up/down: update velocities
    next_x_vel = prev_x_vel
    if action_type == "left" and prev_x_vel != -1:
        next_x_vel = prev_x_vel - 1
    elif action_type == "right" and prev_x_vel != 1:
        next_x_vel = prev_x_vel + 1

    next_y_vel = prev_y_vel
    if action_type == "up" and prev_y_vel != -1:
        next_y_vel = prev_y_vel - 1
    elif action_type == "down" and prev_y_vel != 1:
        next_y_vel = prev_y_vel + 1

    # on true: move blobs by previous velocity
    current_blobs = _move_cells(current_blobs, prev_x_vel, prev_y_vel)

    new_state = copy.deepcopy(state)
    new_state["9Ew"] = current_blobs

    next_hidden = {
        "xVel": int(next_x_vel),
        "yVel": int(next_y_vel),
    }
    return new_state, next_hidden
