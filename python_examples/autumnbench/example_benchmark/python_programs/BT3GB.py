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


def _within_bounds(pos: Tuple[int, int], grid_size: int) -> bool:
    return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size


def _is_free_pos(pos: Tuple[int, int], occupied: set[Tuple[int, int]], grid_size: int) -> bool:
    return _within_bounds(pos, grid_size) and pos not in occupied


def _closest_pos(origin: Tuple[int, int], holes: List[Tuple[int, int]]) -> Tuple[int, int]:
    best = holes[0]
    best_dist = (best[0] - origin[0]) ** 2 + (best[1] - origin[1]) ** 2
    for candidate in holes[1:]:
        dist = (candidate[0] - origin[0]) ** 2 + (candidate[1] - origin[1]) ** 2
        if dist <= best_dist:
            best = candidate
            best_dist = dist
    return best


def _unit_vector_obj_pos(origin: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    sx = 0 if dx == 0 else (-1 if dx < 0 else 1)
    sy = 0 if dy == 0 else (-1 if dy < 0 else 1)
    if abs(sx) == 1 and abs(sy) == 1:
        return sx, 0
    return sx, sy


def _next_liquid(
    pos: Tuple[int, int],
    occupied: set[Tuple[int, int]],
    grid_size: int,
) -> Tuple[int, int]:
    x, y = pos
    if y != grid_size - 1 and _is_free_pos((x, y + 1), occupied, grid_size):
        return x, y + 1

    next_row_y = y + 1
    if next_row_y >= grid_size:
        return x, y

    holes: List[Tuple[int, int]] = []
    for hx in range(grid_size):
        hole = (hx, next_row_y)
        above = (hx, next_row_y - 1)
        if _is_free_pos(hole, occupied, grid_size) and _is_free_pos(above, occupied, grid_size):
            holes.append(hole)

    if not holes:
        return x, y

    closest_hole = _closest_pos((x, y), holes)
    target = (closest_hole[0], closest_hole[1] - 1)
    dx, dy = _unit_vector_obj_pos((x, y), target)
    moved = (x + dx, y + dy)

    if (
        _is_free_pos(target, occupied, grid_size)
        and _is_free_pos(moved, occupied, grid_size)
        and _within_bounds(moved, grid_size)
    ):
        return moved

    return x, y


def _move_down_no_collision(
    pos: Tuple[int, int],
    occupied: set[Tuple[int, int]],
    grid_size: int,
) -> Tuple[int, int]:
    candidate = (pos[0], pos[1] + 1)
    occupied_without_self = set(occupied)
    occupied_without_self.discard(pos)
    if _is_free_pos(candidate, occupied_without_self, grid_size):
        return candidate
    return pos


def _celestial_cells(origin: Tuple[int, int], day: bool) -> List[Dict[str, Any]]:
    ox, oy = origin
    color = "gold" if day else "gray"
    return [
        {"position": {"x": ox + 0, "y": oy + 0}, "color": color},
        {"position": {"x": ox + 0, "y": oy + 1}, "color": color},
        {"position": {"x": ox + 1, "y": oy + 0}, "color": color},
        {"position": {"x": ox + 1, "y": oy + 1}, "color": color},
    ]


def _cloud_cells(origin: Tuple[int, int]) -> List[Dict[str, Any]]:
    ox, oy = origin
    return [
        {"position": {"x": ox - 1, "y": oy}, "color": "gray"},
        {"position": {"x": ox + 0, "y": oy}, "color": "gray"},
        {"position": {"x": ox + 1, "y": oy}, "color": "gray"},
    ]


def _water_cell(x: int, y: int, liquid: bool) -> Dict[str, Any]:
    return {"position": {"x": x, "y": y}, "color": "blue" if liquid else "lightblue"}


def _infer_cloud_origin(cells: List[Dict[str, Any]]) -> Tuple[int, int]:
    positions = [_cell_pos(cell) for cell in cells]
    positions = [pos for pos in positions if pos is not None]
    if not positions:
        return 4, 0
    min_x = min(pos[0] for pos in positions)
    y = positions[0][1]
    return min_x + 1, y


def _infer_day(cells: List[Dict[str, Any]]) -> bool:
    for cell in cells:
        if cell.get("color") == "gold":
            return True
    return False


def _infer_water(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    water = []
    for cell in state.get("water", []):
        pos = _cell_pos(cell)
        if pos is None:
            continue
        color = cell.get("color")
        water.append({"x": pos[0], "y": pos[1], "liquid": color == "blue"})
    return water


def init_state():
    grid_size = 16
    day = True
    celestial_origin = (0, 0)
    cloud_origin = (4, 0)
    water: List[Dict[str, Any]] = []
    state = {
        "GRID_SIZE": grid_size,
        "celestialBody": _celestial_cells(celestial_origin, day),
        "cloud": _cloud_cells(cloud_origin),
        "water": [],
    }
    hidden_state = {
        "day": day,
        "celestial_origin": celestial_origin,
        "cloud_origin": cloud_origin,
        "water": water,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 16), 16)
    if hidden_state is None:
        _, hidden_state = init_state()

    prev_day = hidden_state.get("day")
    if not isinstance(prev_day, bool):
        prev_day = _infer_day(state.get("celestialBody", []))

    prev_cloud_origin = hidden_state.get("cloud_origin")
    if not isinstance(prev_cloud_origin, tuple):
        prev_cloud_origin = _infer_cloud_origin(state.get("cloud", []))

    prev_celestial_origin = hidden_state.get("celestial_origin")
    if not isinstance(prev_celestial_origin, tuple):
        prev_celestial_origin = (0, 0)

    prev_water = hidden_state.get("water")
    if not isinstance(prev_water, list):
        prev_water = _infer_water(state)

    prev_water = copy.deepcopy(prev_water)
    prev_cloud_origin = (int(prev_cloud_origin[0]), int(prev_cloud_origin[1]))

    current_day = prev_day
    current_cloud_origin = prev_cloud_origin
    current_water = copy.deepcopy(prev_water)
    updated_vars: set[str] = set()

    action_type = _action_type(action)

    # --- on left/right ---
    if action_type == "left":
        candidate = (prev_cloud_origin[0] - 1, prev_cloud_origin[1])
        if all(_within_bounds(pos, grid_size) for pos in _positions_set(_cloud_cells(candidate))):
            current_cloud_origin = candidate
        updated_vars.add("cloud")
    elif action_type == "right":
        candidate = (prev_cloud_origin[0] + 1, prev_cloud_origin[1])
        if all(_within_bounds(pos, grid_size) for pos in _positions_set(_cloud_cells(candidate))):
            current_cloud_origin = candidate
        updated_vars.add("cloud")

    # --- on down ---
    if action_type == "down":
        drop_pos = (current_cloud_origin[0], current_cloud_origin[1] + 1)
        current_water.append({"x": drop_pos[0], "y": drop_pos[1], "liquid": current_day})
        updated_vars.add("water")

    # --- on clicked ---
    if action_type == "click":
        current_day = not current_day
        current_water = [
            {"x": drop["x"], "y": drop["y"], "liquid": not drop.get("liquid", False)}
            for drop in current_water
        ]
        updated_vars.add("celestialBody")
        updated_vars.add("water")

    # --- next updates ---
    if "celestialBody" not in updated_vars:
        current_day = prev_day

    if "cloud" not in updated_vars:
        current_cloud_origin = prev_cloud_origin

    if "water" not in updated_vars:
        occupied = _positions_set(_celestial_cells(prev_celestial_origin, prev_day))
        occupied |= _positions_set(_cloud_cells(prev_cloud_origin))
        occupied |= {
            (drop["x"], drop["y"]) for drop in prev_water
        }
        next_water = []
        for drop in prev_water:
            pos = (drop["x"], drop["y"])
            if drop.get("liquid", False):
                new_pos = _next_liquid(pos, occupied, grid_size)
            else:
                new_pos = _move_down_no_collision(pos, occupied, grid_size)
            next_water.append({"x": new_pos[0], "y": new_pos[1], "liquid": drop.get("liquid", False)})
        current_water = next_water

    new_state = {
        "GRID_SIZE": grid_size,
        "celestialBody": _celestial_cells(prev_celestial_origin, current_day),
        "cloud": _cloud_cells(current_cloud_origin),
        "water": [_water_cell(drop["x"], drop["y"], drop.get("liquid", False)) for drop in current_water],
    }

    next_hidden = {
        "day": bool(current_day),
        "celestial_origin": prev_celestial_origin,
        "cloud_origin": current_cloud_origin,
        "water": current_water,
    }
    return new_state, next_hidden
