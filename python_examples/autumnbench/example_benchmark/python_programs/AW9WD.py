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


def _intersects_pos(pos: Tuple[int, int], cells: List[Dict[str, Any]]) -> bool:
    return pos in _positions_set(cells)


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
    egg: Dict[str, Any],
    occupied: set[Tuple[int, int]],
    grid_size: int,
) -> Dict[str, Any]:
    x = _safe_int(egg.get("x"))
    y = _safe_int(egg.get("y"))
    if y != grid_size - 1 and _is_free_pos((x, y + 1), occupied, grid_size):
        return {"x": x, "y": y + 1, "broken": True}

    next_row_y = y + 1
    if next_row_y >= grid_size:
        return {"x": x, "y": y, "broken": True}

    holes: List[Tuple[int, int]] = []
    for hx in range(grid_size):
        hole = (hx, next_row_y)
        above = (hx, next_row_y - 1)
        if _is_free_pos(hole, occupied, grid_size) and _is_free_pos(above, occupied, grid_size):
            holes.append(hole)

    if not holes:
        return {"x": x, "y": y, "broken": True}

    closest_hole = _closest_pos((x, y), holes)
    target = (closest_hole[0], closest_hole[1] - 1)
    dx, dy = _unit_vector_obj_pos((x, y), target)
    moved = (x + dx, y + dy)

    if (
        _is_free_pos(target, occupied, grid_size)
        and _is_free_pos(moved, occupied, grid_size)
        and _within_bounds(moved, grid_size)
    ):
        return {"x": moved[0], "y": moved[1], "broken": True}

    return {"x": x, "y": y, "broken": True}


def _make_eggshells() -> List[Dict[str, Any]]:
    positions = [
        (7, 15), (8, 15), (9, 15),
        (6, 14), (7, 14), (8, 14), (9, 14), (10, 14),
        (5, 13), (6, 13), (7, 13), (8, 13), (9, 13), (10, 13), (11, 13),
        (5, 12), (6, 12), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12),
        (5, 11), (6, 11), (7, 11), (8, 11), (9, 11), (10, 11), (11, 11),
        (6, 10), (7, 10), (8, 10), (9, 10), (10, 10),
        (7, 9), (8, 9), (9, 9),
    ]
    return [{"x": x, "y": y, "broken": False} for x, y in positions]


def _make_feathers() -> List[Dict[str, Any]]:
    data = [
        ("orange", 7, 15),
        ("orange", 8, 15),
        ("yellow", 6, 14),
        ("yellow", 7, 14),
        ("yellow", 8, 14),
        ("yellow", 9, 14),
        ("yellow", 6, 13),
        ("yellow", 7, 13),
        ("yellow", 8, 13),
        ("yellow", 9, 13),
        ("yellow", 6, 12),
        ("yellow", 7, 12),
        ("yellow", 8, 12),
        ("yellow", 9, 12),
        ("yellow", 9, 11),
        ("yellow", 10, 11),
        ("orange", 11, 11),
        ("yellow", 9, 10),
    ]
    return [
        {"position": {"x": x, "y": y}, "color": color}
        for color, x, y in data
    ]


def _render_eggshells(eggshells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rendered = []
    for egg in eggshells:
        rendered.append(
            {
                "position": {"x": _safe_int(egg.get("x")), "y": _safe_int(egg.get("y"))},
                "color": "tan",
            }
        )
    return rendered


def _hidden_from_state(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    eggshells = []
    for cell in state.get("eggshells", []) if isinstance(state, dict) else []:
        pos = _cell_pos(cell)
        if pos is None:
            continue
        eggshells.append({"x": pos[0], "y": pos[1], "broken": False})
    return eggshells


def init_state():
    grid_size = 16
    feathers = _make_feathers()
    eggshells = _make_eggshells()
    state = {
        "GRID_SIZE": grid_size,
        "feathers": feathers,
        "eggshells": _render_eggshells(eggshells),
    }
    hidden_state = {
        "eggshells": eggshells,
        "feathers": feathers,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 16), 16)

    if hidden_state is None:
        _, hidden_state = init_state()

    prev_eggshells = hidden_state.get("eggshells")
    if not isinstance(prev_eggshells, list):
        prev_eggshells = _hidden_from_state(state)
    prev_feathers = hidden_state.get("feathers")
    if not isinstance(prev_feathers, list):
        prev_feathers = copy.deepcopy(state.get("feathers", []))

    prev_feathers = copy.deepcopy(prev_feathers)
    prev_eggshells = copy.deepcopy(prev_eggshells)

    updated_eggshells = prev_eggshells
    updated = False

    action_type = _action_type(action)
    click_pos = None
    if action_type == "click" and isinstance(action, dict):
        click_pos = (_safe_int(action.get("x")), _safe_int(action.get("y")))

    if click_pos is not None:
        if any((egg["x"], egg["y"]) == click_pos for egg in prev_eggshells):
            updated = True
            updated_eggshells = []
            for egg in prev_eggshells:
                if (egg["x"], egg["y"]) == click_pos:
                    updated_eggshells.append({"x": egg["x"], "y": egg["y"], "broken": True})
                else:
                    updated_eggshells.append(copy.deepcopy(egg))

    if not updated:
        broken = [egg for egg in prev_eggshells if egg.get("broken")]
        broken = [
            egg for egg in broken
            if not _intersects_pos((egg["x"], egg["y"]), prev_feathers)
        ]
        unbroken = [egg for egg in prev_eggshells if not egg.get("broken")]

        occupied = _positions_set(_render_eggshells(prev_eggshells)) | _positions_set(prev_feathers)
        next_broken = [_next_liquid(egg, occupied, grid_size) for egg in broken]
        updated_eggshells = next_broken + unbroken

    new_state = {
        "GRID_SIZE": grid_size,
        "feathers": copy.deepcopy(prev_feathers),
        "eggshells": _render_eggshells(updated_eggshells),
    }

    next_hidden = {
        "eggshells": updated_eggshells,
        "feathers": prev_feathers,
    }
    return new_state, next_hidden
