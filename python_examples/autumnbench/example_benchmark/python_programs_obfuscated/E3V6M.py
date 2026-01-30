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


DIR_VECTORS: Dict[int, Tuple[int, int]] = {
    0: (1, 0),
    1: (1, 1),
    2: (0, 1),
    3: (-1, 1),
    4: (-1, 0),
    5: (-1, -1),
    6: (0, -1),
    7: (1, -1),
}
VECTOR_DIR = {v: k for k, v in DIR_VECTORS.items()}


def _render_lights(origin: Tuple[int, int], direction: int, turned_on: bool) -> List[Dict[str, Any]]:
    dx, dy = DIR_VECTORS.get(direction, (1, 0))
    color = "yellow" if turned_on else "white"
    cells = []
    for i in range(-2, 3):
        cells.append({
            "position": {"x": origin[0] + dx * i, "y": origin[1] + dy * i},
            "color": color,
        })
    return cells


def _infer_lights_from_state(
    lights: List[Dict[str, Any]],
    fallback_origin: Tuple[int, int],
    fallback_dir: int,
    fallback_on: bool,
) -> Tuple[Tuple[int, int], int, bool]:
    if not lights:
        return fallback_origin, fallback_dir, fallback_on

    positions = [pos for pos in (_cell_pos(cell) for cell in lights) if pos is not None]
    turned_on = lights[0].get("color") == "yellow" if lights else fallback_on
    if not positions:
        return fallback_origin, fallback_dir, turned_on

    min_x = min(p[0] for p in positions)
    max_x = max(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] for p in positions)
    origin = ((min_x + max_x) // 2, (min_y + max_y) // 2)

    xs = {p[0] for p in positions}
    ys = {p[1] for p in positions}
    diffs = {p[0] - p[1] for p in positions}
    sums = {p[0] + p[1] for p in positions}

    possible_dirs: List[int] = []
    if len(xs) == 1:
        possible_dirs = [2, 6]
    elif len(ys) == 1:
        possible_dirs = [0, 4]
    elif len(diffs) == 1:
        possible_dirs = [1, 5]
    elif len(sums) == 1:
        possible_dirs = [3, 7]

    if possible_dirs:
        direction = fallback_dir if fallback_dir in possible_dirs else possible_dirs[0]
    else:
        direction = fallback_dir
    return origin, direction, turned_on


def init_state():
    grid_size = 16
    origin = (7, 7)
    direction = 1
    turned_on = False
    state = {
        "GRID_SIZE": grid_size,
        "G7F": _render_lights(origin, direction, turned_on),
    }
    hidden_state = {
        "origin": origin,
        "dir": direction,
        "turnedOn": turned_on,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 16), 16)
    lights = copy.deepcopy(state.get("G7F", []))

    if hidden_state is None:
        _, hidden_state = init_state()

    fallback_origin = hidden_state.get("origin", (7, 7))
    fallback_dir = int(hidden_state.get("dir", 1))
    fallback_on = bool(hidden_state.get("turnedOn", False))

    origin, direction, turned_on = _infer_lights_from_state(
        lights,
        fallback_origin,
        fallback_dir,
        fallback_on,
    )

    action_type = _action_type(action)

    # --- on clicked ---
    if action_type == "click":
        turned_on = not turned_on

    # --- move if turned off ---
    if action_type in {"up", "down", "left", "right"} and not turned_on:
        dx, dy = 0, 0
        if action_type == "up":
            dy = -1
        elif action_type == "down":
            dy = 1
        elif action_type == "left":
            dx = -1
        elif action_type == "right":
            dx = 1
        origin = (origin[0] + dx, origin[1] + dy)

    # --- rotate if turned on ---
    if action_type in {"up", "down", "left", "right"} and turned_on:
        if action_type == "up":
            direction = (direction + 1) % 8
        elif action_type == "down":
            direction = (direction + 7) % 8
        elif action_type == "left":
            direction = (direction + 2) % 8
        elif action_type == "right":
            direction = (direction + 6) % 8

    new_state = {
        "GRID_SIZE": grid_size,
        "G7F": _render_lights(origin, direction, turned_on),
    }
    next_hidden = {
        "origin": origin,
        "dir": int(direction),
        "turnedOn": bool(turned_on),
    }
    return new_state, next_hidden
