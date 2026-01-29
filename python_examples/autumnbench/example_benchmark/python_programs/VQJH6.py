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


def _blob_cells(origin: Tuple[int, int]) -> List[Dict[str, Any]]:
    ox, oy = origin
    return [
        _make_cell(ox, oy - 1, "blue"),
        _make_cell(ox, oy, "blue"),
        _make_cell(ox + 1, oy - 1, "blue"),
        _make_cell(ox + 1, oy, "blue"),
    ]


def _infer_blob_origins(cells: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    positions = _positions_set(cells)
    origins: set[Tuple[int, int]] = set()
    for x, y in positions:
        if (
            (x, y) in positions
            and (x + 1, y) in positions
            and (x, y - 1) in positions
            and (x + 1, y - 1) in positions
        ):
            origins.add((x, y))
    return sorted(origins)


def init_state():
    grid_size = 17
    mid = grid_size // 2
    left_button = [_make_cell(0, mid, "red")]
    right_button = [_make_cell(grid_size - 1, mid, "darkorange")]
    up_button = [_make_cell(mid, 0, "gold")]
    down_button = [_make_cell(mid, grid_size - 1, "green")]

    state = {
        "GRID_SIZE": grid_size,
        "leftButton": left_button,
        "rightButton": right_button,
        "upButton": up_button,
        "downButton": down_button,
        "blobs": [],
    }
    hidden_state = {
        "gravity": "down",
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    if hidden_state is None:
        hidden_state = {}

    grid_size = _safe_int(state.get("GRID_SIZE", 17), 17)
    left_button = copy.deepcopy(state.get("leftButton", []))
    right_button = copy.deepcopy(state.get("rightButton", []))
    up_button = copy.deepcopy(state.get("upButton", []))
    down_button = copy.deepcopy(state.get("downButton", []))

    prev_blob_cells = copy.deepcopy(state.get("blobs", []))
    blob_origins = _infer_blob_origins(prev_blob_cells)

    gravity = str(hidden_state.get("gravity", "down"))
    dx, dy = 0, 0
    if gravity == "left":
        dx, dy = -1, 0
    elif gravity == "right":
        dx, dy = 1, 0
    elif gravity == "up":
        dx, dy = 0, -1
    elif gravity == "down":
        dx, dy = 0, 1

    moved_origins = [(x + dx, y + dy) for x, y in blob_origins]

    action_type = _action_type(action)
    click_pos = None
    if action_type == "click" and isinstance(action, dict):
        click_pos = (_safe_int(action.get("x")), _safe_int(action.get("y")))

    # on clicked add blob if free (after movement)
    if click_pos is not None:
        occupied = (
            _positions_set([cell for origin in moved_origins for cell in _blob_cells(origin)])
            | _positions_set(left_button)
            | _positions_set(right_button)
            | _positions_set(up_button)
            | _positions_set(down_button)
        )
        if click_pos not in occupied:
            moved_origins.append(click_pos)

    # on clicked buttons update gravity (after adding)
    if click_pos is not None:
        left_pos = _cell_pos(left_button[0]) if left_button else None
        right_pos = _cell_pos(right_button[0]) if right_button else None
        up_pos = _cell_pos(up_button[0]) if up_button else None
        down_pos = _cell_pos(down_button[0]) if down_button else None

        if left_pos is not None and click_pos == left_pos:
            gravity = "left"
        if right_pos is not None and click_pos == right_pos:
            gravity = "right"
        if up_pos is not None and click_pos == up_pos:
            gravity = "up"
        if down_pos is not None and click_pos == down_pos:
            gravity = "down"

    blobs_cells: List[Dict[str, Any]] = []
    for origin in moved_origins:
        blobs_cells.extend(_blob_cells(origin))

    new_state = {
        "GRID_SIZE": grid_size,
        "leftButton": left_button,
        "rightButton": right_button,
        "upButton": up_button,
        "downButton": down_button,
        "blobs": blobs_cells,
    }
    next_hidden = {
        "gravity": gravity,
    }
    return new_state, next_hidden
