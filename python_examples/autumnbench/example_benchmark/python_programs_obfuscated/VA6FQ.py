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


def _sign(value: int) -> int:
    if value == 0:
        return 0
    return -1 if value < 0 else 1


def _unit_vector_obj_pos(pos: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
    delta_x = target[0] - pos[0]
    delta_y = target[1] - pos[1]
    sign_x = _sign(delta_x)
    sign_y = _sign(delta_y)
    if abs(sign_x) == 1 and abs(sign_y) == 1:
        return sign_x, 0
    return sign_x, sign_y


def _within_bounds(pos: Tuple[int, int], grid_size: int) -> bool:
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size


def _is_free_pos(pos: Tuple[int, int], occupied: set[Tuple[int, int]]) -> bool:
    return pos not in occupied


def _next_solid(pos: Tuple[int, int], grid_size: int, occupied: set[Tuple[int, int]]) -> Tuple[int, int]:
    below = (pos[0], pos[1] + 1)
    if _within_bounds(below, grid_size) and _is_free_pos(below, occupied):
        return below
    return pos


def _next_liquid(pos: Tuple[int, int], grid_size: int, occupied: set[Tuple[int, int]]) -> Tuple[int, int]:
    if pos[1] != grid_size - 1:
        below = (pos[0], pos[1] + 1)
        if _is_free_pos(below, occupied):
            return below

    next_row_y = pos[1] + 1
    if next_row_y >= grid_size:
        return pos

    next_row = [(x, next_row_y) for x in range(grid_size)]
    holes = [
        p for p in next_row
        if _is_free_pos(p, occupied) and _is_free_pos((p[0], p[1] - 1), occupied)
    ]
    if not holes:
        return pos

    closest_hole = None
    best_dist = None
    for hole in holes:
        dist = (hole[0] - pos[0]) ** 2 + (hole[1] - pos[1]) ** 2
        if best_dist is None or dist <= best_dist:
            best_dist = dist
            closest_hole = hole
    if closest_hole is None:
        return pos
    target = (closest_hole[0], closest_hole[1] - 1)
    dir_x, dir_y = _unit_vector_obj_pos(pos, target)
    moved = (pos[0] + dir_x, pos[1] + dir_y)

    if _within_bounds(moved, grid_size) and _is_free_pos(moved, occupied):
        return moved
    return pos


def init_state():
    grid_size = 10
    sand_button = [_make_cell(2, 0, "red")]
    water_button = [_make_cell(7, 0, "green")]

    sand_positions = [
        (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9),
        (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8),
        (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
        (2, 6), (4, 6), (5, 6), (7, 6),
        (2, 5), (4, 5), (5, 5), (7, 5),
    ]
    sand = [_make_cell(x, y, "tan") for x, y in sand_positions]

    state = {
        "GRID_SIZE": grid_size,
        "S71": sand_button,
        "FQP": water_button,
        "1f5": sand,
        "LrC": [],
    }
    hidden_state = {
        "clickType": "1f5",
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    if hidden_state is None:
        hidden_state = {}

    grid_size = _safe_int(state.get("GRID_SIZE", 10), 10)
    prev_sand = copy.deepcopy(state.get("1f5", []))
    prev_water = copy.deepcopy(state.get("LrC", []))
    sand_button = copy.deepcopy(state.get("S71", []))
    water_button = copy.deepcopy(state.get("FQP", []))
    click_type = str(hidden_state.get("clickType", "1f5"))

    occupied_prev = (
        _positions_set(prev_sand)
        | _positions_set(prev_water)
        | _positions_set(sand_button)
        | _positions_set(water_button)
    )
    prev_water_positions = _positions_set(prev_water)

    # on true: update sand based on prev water
    current_sand: List[Dict[str, Any]] = []
    for cell in prev_sand:
        pos = _cell_pos(cell)
        if pos is None:
            continue
        liquid = cell.get("color") == "sandybrown"
        neighbors = [
            (pos[0] - 1, pos[1]),
            (pos[0] + 1, pos[1]),
            (pos[0], pos[1] - 1),
            (pos[0], pos[1] + 1),
        ]
        if any(npos in prev_water_positions for npos in neighbors):
            liquid = True

        if liquid:
            next_pos = _next_liquid(pos, grid_size, occupied_prev)
            color = "sandybrown"
        else:
            next_pos = _next_solid(pos, grid_size, occupied_prev)
            color = "tan"
        current_sand.append(_make_cell(next_pos[0], next_pos[1], color))

    action_type = _action_type(action)
    click_pos = None
    if action_type == "click" and isinstance(action, dict):
        click_pos = (_safe_int(action.get("x")), _safe_int(action.get("y")))

    # on clicked sandButton / waterButton
    if click_pos is not None:
        sand_button_pos = _cell_pos(sand_button[0]) if sand_button else None
        if sand_button_pos is not None and click_pos == sand_button_pos:
            click_type = "1f5"

        water_button_pos = _cell_pos(water_button[0]) if water_button else None
        if water_button_pos is not None and click_pos == water_button_pos:
            click_type = "LrC"

    current_water = copy.deepcopy(prev_water)
    water_updated = False

    # on clicked add sand/water (uses updated sand)
    if click_pos is not None:
        if _is_free_pos(click_pos, occupied_prev):
            if click_type == "1f5":
                current_sand.append(_make_cell(click_pos[0], click_pos[1], "tan"))
            elif click_type == "LrC":
                current_water.append(_make_cell(click_pos[0], click_pos[1], "skyblue"))
                water_updated = True

    # initnext for water if not updated by click
    if not water_updated:
        current_water = []
        for cell in prev_water:
            pos = _cell_pos(cell)
            if pos is None:
                continue
            next_pos = _next_liquid(pos, grid_size, occupied_prev)
            current_water.append(_make_cell(next_pos[0], next_pos[1], "skyblue"))

    new_state = {
        "GRID_SIZE": grid_size,
        "S71": sand_button,
        "FQP": water_button,
        "1f5": current_sand,
        "LrC": current_water,
    }
    next_hidden = {
        "clickType": click_type,
    }
    return new_state, next_hidden
