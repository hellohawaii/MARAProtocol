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


def _pos_pole(cells: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return cells[0] if cells else None


def _neg_pole(cells: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    return cells[-1] if cells else None


def _adjacent_elem(elem1: Dict[str, Any] | None, elem2: Dict[str, Any] | None) -> bool:
    if not elem1 or not elem2:
        return False
    pos1 = _cell_pos(elem1)
    pos2 = _cell_pos(elem2)
    if pos1 is None or pos2 is None:
        return False
    return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1]) == 1


def _delta_elem(elem1: Dict[str, Any] | None, elem2: Dict[str, Any] | None) -> Tuple[int, int] | None:
    if not elem1 or not elem2:
        return None
    pos1 = _cell_pos(elem1)
    pos2 = _cell_pos(elem2)
    if pos1 is None or pos2 is None:
        return None
    return pos2[0] - pos1[0], pos2[1] - pos1[1]


def _unit_vector_single_pos(delta: Tuple[int, int]) -> Tuple[int, int]:
    def _sign(v: int) -> int:
        if v == 0:
            return 0
        return -1 if v < 0 else 1

    return _sign(delta[0]), _sign(delta[1])


def _move_no_collision(
    prev_cells: List[Dict[str, Any]],
    dx: int,
    dy: int,
    fixed_cells: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    moved_cells = _shift_cells(prev_cells, dx, dy)
    prev_positions = _positions_set(prev_cells)
    moved_positions = _positions_set(moved_cells)
    fixed_positions = _positions_set(fixed_cells)
    for pos in moved_positions - prev_positions:
        if pos in fixed_positions:
            return copy.deepcopy(prev_cells)
    return moved_cells


def init_state():
    grid_size = 16
    fixed_magnet = [
        {"position": {"x": 7, "y": 7}, "color": "red"},
        {"position": {"x": 7, "y": 8}, "color": "red"},
    ]
    mobile_magnet = [
        {"position": {"x": 4, "y": 7}, "color": "blue"},
        {"position": {"x": 4, "y": 8}, "color": "blue"},
    ]
    state = {
        "GRID_SIZE": grid_size,
        "fixedMagnet": fixed_magnet,
        "mobileMagnet": mobile_magnet,
    }
    return state, {}


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    new_state = copy.deepcopy(state)
    if hidden_state is None:
        hidden_state = {}

    prev_mobile = copy.deepcopy(state.get("mobileMagnet", []))
    fixed_magnet = copy.deepcopy(state.get("fixedMagnet", []))

    action_type = _action_type(action)

    # On-clauses: action-controlled movement (uses prev mobile magnet).
    if action_type == "left":
        new_state["mobileMagnet"] = _move_no_collision(prev_mobile, -1, 0, fixed_magnet)
    elif action_type == "right":
        new_state["mobileMagnet"] = _move_no_collision(prev_mobile, 1, 0, fixed_magnet)
    elif action_type == "up":
        new_state["mobileMagnet"] = _move_no_collision(prev_mobile, 0, -1, fixed_magnet)
    elif action_type == "down":
        new_state["mobileMagnet"] = _move_no_collision(prev_mobile, 0, 1, fixed_magnet)

    current_mobile = new_state.get("mobileMagnet", [])

    # Repulsion: like poles adjacent reset to previous step.
    if _adjacent_elem(_pos_pole(current_mobile), _pos_pole(fixed_magnet)):
        new_state["mobileMagnet"] = copy.deepcopy(prev_mobile)

    current_mobile = new_state.get("mobileMagnet", [])
    if _adjacent_elem(_neg_pole(current_mobile), _neg_pole(fixed_magnet)):
        new_state["mobileMagnet"] = copy.deepcopy(prev_mobile)

    current_mobile = new_state.get("mobileMagnet", [])

    # Attraction: opposite poles two units apart pull by one step.
    attract_vectors = {(0, 2), (2, 0), (-2, 0), (0, -2)}
    delta = _delta_elem(_pos_pole(current_mobile), _neg_pole(fixed_magnet))
    if delta is not None and delta in attract_vectors:
        dx, dy = _unit_vector_single_pos(delta)
        new_state["mobileMagnet"] = _shift_cells(current_mobile, dx, dy)

    current_mobile = new_state.get("mobileMagnet", [])
    delta = _delta_elem(_neg_pole(current_mobile), _pos_pole(fixed_magnet))
    if delta is not None and delta in attract_vectors:
        dx, dy = _unit_vector_single_pos(delta)
        new_state["mobileMagnet"] = _shift_cells(current_mobile, dx, dy)

    return new_state, hidden_state
