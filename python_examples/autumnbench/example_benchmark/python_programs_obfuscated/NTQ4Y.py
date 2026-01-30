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


def _button_cells(x: int, y: int, color: str) -> List[Dict[str, Any]]:
    return [{"position": {"x": x, "y": y}, "color": color}]


def _occupied_from_parts(parts: List[List[Dict[str, Any]]]) -> set[Tuple[int, int]]:
    occupied: set[Tuple[int, int]] = set()
    for cells in parts:
        occupied |= _positions_set(cells)
    return occupied


def _clicked_obj(action: Any, obj_cells: List[Dict[str, Any]]) -> bool:
    if _action_type(action) != "click" or not isinstance(action, dict):
        return False
    click_pos = (_safe_int(action.get("x")), _safe_int(action.get("y")))
    return click_pos in _positions_set(obj_cells)


def init_state():
    grid_size = 16
    vessels_positions = [
        (6, 15), (6, 14), (6, 13), (5, 12), (4, 11), (3, 10),
        (9, 15), (9, 14), (9, 13), (10, 12), (11, 11), (12, 10),
    ]
    plugs_positions = [(7, 15), (8, 15), (7, 14), (8, 14), (7, 13), (8, 13)]

    state = {
        "GRID_SIZE": grid_size,
        "5RL": _button_cells(2, 0, "purple"),
        "lm0": _button_cells(5, 0, "orange"),
        "FQP": _button_cells(8, 0, "blue"),
        "QqE": _button_cells(11, 0, "gray"),
        "5jt": _button_cells(14, 0, "red"),
        "08A": [{"position": {"x": x, "y": y}, "color": "purple"} for x, y in vessels_positions],
        "lcN": [{"position": {"x": x, "y": y}, "color": "orange"} for x, y in plugs_positions],
        "S6Y": [],
    }

    hidden_state = {"currentParticle": "vessel"}
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 16), 16)
    if hidden_state is None:
        hidden_state = {}

    prev_vessels = copy.deepcopy(state.get("08A", []))
    prev_plugs = copy.deepcopy(state.get("lcN", []))
    prev_water = copy.deepcopy(state.get("S6Y", []))

    vessel_button = copy.deepcopy(state.get("5RL", _button_cells(2, 0, "purple")))
    plug_button = copy.deepcopy(state.get("lm0", _button_cells(5, 0, "orange")))
    water_button = copy.deepcopy(state.get("FQP", _button_cells(8, 0, "blue")))
    remove_button = copy.deepcopy(state.get("QqE", _button_cells(11, 0, "gray")))
    clear_button = copy.deepcopy(state.get("5jt", _button_cells(14, 0, "red")))

    current_particle = str(hidden_state.get("currentParticle", "vessel"))
    updated_vars: set[str] = set()

    # --- on true: update waterList ---
    prev_occupied = _occupied_from_parts(
        [prev_vessels, prev_plugs, prev_water, vessel_button, plug_button, water_button, remove_button, clear_button]
    )
    current_water = []
    for cell in prev_water:
        pos = _cell_pos(cell)
        if pos is None:
            continue
        next_pos = _next_liquid(pos, prev_occupied, grid_size)
        current_water.append({"position": {"x": next_pos[0], "y": next_pos[1]}, "color": "blue"})
    updated_vars.add("S6Y")

    current_vessels = copy.deepcopy(prev_vessels)
    current_plugs = copy.deepcopy(prev_plugs)

    # --- add objects based on currentParticle ---
    if _action_type(action) == "click" and isinstance(action, dict):
        click_pos = (_safe_int(action.get("x")), _safe_int(action.get("y")))
        current_occupied = _occupied_from_parts(
            [current_vessels, current_plugs, current_water, vessel_button, plug_button, water_button, remove_button, clear_button]
        )

        if _is_free_pos(click_pos, current_occupied, grid_size) and current_particle == "vessel":
            current_vessels.append({"position": {"x": click_pos[0], "y": click_pos[1]}, "color": "purple"})
            updated_vars.add("08A")

        if _is_free_pos(click_pos, current_occupied, grid_size) and current_particle == "plug":
            current_plugs.append({"position": {"x": click_pos[0], "y": click_pos[1]}, "color": "orange"})
            updated_vars.add("lcN")

        if _is_free_pos(click_pos, current_occupied, grid_size) and current_particle == "LrC":
            current_water.append({"position": {"x": click_pos[0], "y": click_pos[1]}, "color": "blue"})
            updated_vars.add("S6Y")

    # --- button clicks update currentParticle ---
    if _clicked_obj(action, vessel_button):
        current_particle = "vessel"
        updated_vars.add("currentParticle")

    if _clicked_obj(action, plug_button):
        current_particle = "plug"
        updated_vars.add("currentParticle")

    if _clicked_obj(action, water_button):
        current_particle = "LrC"
        updated_vars.add("currentParticle")

    # --- remove and clear buttons ---
    if _clicked_obj(action, remove_button):
        current_plugs = []
        updated_vars.add("lcN")

    if _clicked_obj(action, clear_button):
        current_vessels = []
        current_plugs = []
        current_water = []
        updated_vars.update({"08A", "lcN", "S6Y"})

    next_state = {
        "GRID_SIZE": grid_size,
        "5RL": vessel_button,
        "lm0": plug_button,
        "FQP": water_button,
        "QqE": remove_button,
        "5jt": clear_button,
        "08A": current_vessels,
        "lcN": current_plugs,
        "S6Y": current_water,
    }

    next_hidden = {"currentParticle": current_particle}
    return next_state, next_hidden
