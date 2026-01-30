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


def _collect_occupied(state: Dict[str, Any]) -> set[Tuple[int, int]]:
    occupied: set[Tuple[int, int]] = set()
    for key, value in state.items():
        if key == "GRID_SIZE":
            continue
        if not isinstance(value, list):
            continue
        occupied |= _positions_set(value)
    return occupied


def _within_bounds_pos(pos: Tuple[int, int], grid_size: int) -> bool:
    x, y = pos
    return 0 <= x < grid_size and 0 <= y < grid_size


def _is_free_pos(pos: Tuple[int, int], occupied: set[Tuple[int, int]], grid_size: int) -> bool:
    return _within_bounds_pos(pos, grid_size) and pos not in occupied


def _sqdist(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return dx * dx + dy * dy


def _closest_pos(origin: Tuple[int, int], positions: List[Tuple[int, int]]) -> Tuple[int, int] | None:
    if not positions:
        return None
    best = positions[0]
    best_dist = _sqdist(origin, best)
    for pos in positions[1:]:
        dist = _sqdist(origin, pos)
        if dist <= best_dist:
            best = pos
            best_dist = dist
    return best


def _unit_vector_obj_pos(origin: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]

    def _sign(v: int) -> int:
        if v == 0:
            return 0
        return -1 if v < 0 else 1

    sign_x = _sign(dx)
    sign_y = _sign(dy)
    if abs(sign_x) == 1 and abs(sign_y) == 1:
        return sign_x, 0
    return sign_x, sign_y


def _next_liquid(
    pos: Tuple[int, int],
    occupied: set[Tuple[int, int]],
    grid_size: int,
) -> Tuple[int, int]:
    x, y = pos
    below = (x, y + 1)
    if y != grid_size - 1 and _is_free_pos(below, occupied, grid_size):
        return below

    next_row_y = y + 1
    if next_row_y >= grid_size:
        return pos

    next_row = [(x2, next_row_y) for x2 in range(grid_size)]
    holes = [
        p for p in next_row
        if _is_free_pos(p, occupied, grid_size)
        and _is_free_pos((p[0], p[1] - 1), occupied, grid_size)
    ]
    closest = _closest_pos(pos, holes)
    if closest is None:
        return pos

    target = (closest[0], closest[1] - 1)
    dx, dy = _unit_vector_obj_pos(pos, target)
    moved = (pos[0] + dx, pos[1] + dy)
    if (
        _is_free_pos(target, occupied, grid_size)
        and _is_free_pos(moved, occupied, grid_size)
        and _within_bounds_pos(moved, grid_size)
    ):
        return moved
    return pos


def _render_crate(origin: Tuple[int, int]) -> List[Dict[str, Any]]:
    ox, oy = origin
    offsets = [
        (-2, -1), (2, -1),
        (-2, 0), (2, 0),
        (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
    ]
    return [
        {"position": {"x": ox + dx, "y": oy + dy}, "color": "brown"}
        for dx, dy in offsets
    ]


def _infer_crate_origin(cells: List[Dict[str, Any]]) -> Tuple[int, int] | None:
    positions = [_cell_pos(cell) for cell in cells]
    positions = [pos for pos in positions if pos is not None]
    if not positions:
        return None
    min_x = min(pos[0] for pos in positions)
    min_y = min(pos[1] for pos in positions)
    return min_x + 2, min_y + 1


def _in_crate(rock_pos: Tuple[int, int], crate_origin: Tuple[int, int]) -> bool:
    rock_x, rock_y = rock_pos
    crate_x, crate_y = crate_origin
    return (
        rock_x >= crate_x - 2
        and rock_x < crate_x + 3
        and rock_y >= 0
        and rock_y < crate_y + 1
    )


def init_state():
    grid_size = 7
    water_cells = [
        {"position": {"x": x, "y": y}, "color": "blue"}
        for y in range(3, grid_size)
        for x in range(grid_size)
    ]

    crate_origin = (3, 1)
    state = {
        "GRID_SIZE": grid_size,
        "LrC": water_cells,
        "Wi4": [],
        "vEJ": _render_crate(crate_origin),
    }

    hidden_state = {
        "crate_origin": crate_origin,
        "crate_add_weight": 0,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 7), 7)
    if hidden_state is None:
        hidden_state = {}

    prev_water_cells = copy.deepcopy(state.get("LrC", []))
    prev_rocks_cells = copy.deepcopy(state.get("Wi4", []))
    prev_crate_cells = copy.deepcopy(state.get("vEJ", []))

    prev_crate_origin = _infer_crate_origin(prev_crate_cells)
    if prev_crate_origin is None:
        prev_crate_origin = hidden_state.get("crate_origin", (3, 1))
    prev_add_weight = int(hidden_state.get("crate_add_weight", 0))

    prev_rock_positions = [_cell_pos(cell) for cell in prev_rocks_cells]
    prev_rock_positions = [pos for pos in prev_rock_positions if pos is not None]

    prev_occupied = _collect_occupied(state)
    action_type = _action_type(action)

    current_water_cells = copy.deepcopy(prev_water_cells)
    current_rocks_cells = copy.deepcopy(prev_rocks_cells)
    current_crate_origin = prev_crate_origin
    current_add_weight = prev_add_weight
    updated_vars: set[str] = set()

    # --- on clicked: add rock if free ---
    if action_type == "click" and isinstance(action, dict):
        click_pos = (_safe_int(action.get("x")), _safe_int(action.get("y")))
        if _is_free_pos(click_pos, prev_occupied, grid_size):
            current_rocks_cells = copy.deepcopy(prev_rocks_cells)
            current_rocks_cells.append(
                {"position": {"x": click_pos[0], "y": click_pos[1]}, "color": "silver"}
            )
            updated_vars.add("Wi4")

    # --- next updates ---
    if "vEJ" not in updated_vars:
        rocks_in_crate = [pos for pos in prev_rock_positions if _in_crate(pos, prev_crate_origin)]
        weight = len(rocks_in_crate)
        if prev_add_weight < weight and weight < 5:
            current_add_weight = prev_add_weight + 1
            current_crate_origin = (prev_crate_origin[0], prev_crate_origin[1] + 1)
        else:
            current_add_weight = prev_add_weight
            current_crate_origin = prev_crate_origin

    if "Wi4" not in updated_vars:
        crate_positions = _positions_set(_render_crate(current_crate_origin))
        next_rocks_cells = []
        for cell in prev_rocks_cells:
            pos = _cell_pos(cell)
            if pos is None:
                continue
            next_pos = (pos[0], pos[1] + 1)
            no_obj_below = next_pos not in crate_positions and next_pos not in prev_rock_positions
            if _within_bounds_pos(next_pos, grid_size) and no_obj_below:
                pos = next_pos
            next_rocks_cells.append({"position": {"x": pos[0], "y": pos[1]}, "color": "silver"})
        current_rocks_cells = next_rocks_cells

    rocks_for_water = current_rocks_cells
    rock_positions_for_water = _positions_set(rocks_for_water)

    if "LrC" not in updated_vars:
        current_water_cells = []
        for cell in prev_water_cells:
            pos = _cell_pos(cell)
            if pos is None:
                continue
            if pos in rock_positions_for_water:
                next_pos = (pos[0], pos[1] - 1)
            else:
                next_pos = _next_liquid(pos, prev_occupied, grid_size)
            current_water_cells.append({"position": {"x": next_pos[0], "y": next_pos[1]}, "color": "blue"})

    next_state = {
        "GRID_SIZE": grid_size,
        "LrC": current_water_cells,
        "Wi4": current_rocks_cells,
        "vEJ": _render_crate(current_crate_origin),
    }

    next_hidden = {
        "crate_origin": current_crate_origin,
        "crate_add_weight": int(current_add_weight),
    }
    return next_state, next_hidden
