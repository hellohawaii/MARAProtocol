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


def _intersects(cells_a: List[Dict[str, Any]], cells_b: List[Dict[str, Any]]) -> bool:
    if not cells_a or not cells_b:
        return False
    return bool(_positions_set(cells_a) & _positions_set(cells_b))


def _make_cell(x: int, y: int, color: str) -> Dict[str, Any]:
    return {"position": {"x": x, "y": y}, "color": color}


def init_state():
    grid_size = 16
    agent = [_make_cell(7, 9, "red")]

    coins = []
    for y in range(2, 5):
        for x in range(3, 13):
            if x % 2 == 0 and y % 2 == 0:
                coins.append(_make_cell(x, y, "gold"))

    state = {
        "GRID_SIZE": grid_size,
        "agent": agent,
        "coins": coins,
        "bullets": [],
    }
    hidden_state = {
        "numBullets": 0,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    if hidden_state is None:
        hidden_state = {}

    prev_agent = copy.deepcopy(state.get("agent", []))
    prev_coins = copy.deepcopy(state.get("coins", []))
    prev_bullets = copy.deepcopy(state.get("bullets", []))
    num_bullets = int(hidden_state.get("numBullets", 0))

    action_type = _action_type(action)

    # on left/right/up/down: move agent (uses prev agent)
    current_agent = copy.deepcopy(prev_agent)
    if action_type == "left":
        current_agent = _shift_cells(prev_agent, -1, 0)
    elif action_type == "right":
        current_agent = _shift_cells(prev_agent, 1, 0)
    elif action_type == "up":
        current_agent = _shift_cells(prev_agent, 0, -1)
    elif action_type == "down":
        current_agent = _shift_cells(prev_agent, 0, 1)

    # on true: bullets move up (uses prev bullets)
    current_bullets = _shift_cells(prev_bullets, 0, -1)

    # on clicked with bullets: decrement and spawn bullet at prev agent
    if action_type == "click" and isinstance(action, dict) and num_bullets > 0:
        num_bullets -= 1
        agent_pos = _cell_pos(prev_agent[0]) if prev_agent else None
        if agent_pos is not None:
            current_bullets.append(_make_cell(agent_pos[0], agent_pos[1], "mediumpurple"))

    # on intersects (prev agent, prev coins): gain bullet and remove coin
    current_coins = copy.deepcopy(prev_coins)
    if _intersects(prev_agent, prev_coins):
        num_bullets += 1
        current_coins = []
        for coin in prev_coins:
            if not _intersects([coin], prev_agent):
                current_coins.append(copy.deepcopy(coin))

    new_state = {
        "GRID_SIZE": state.get("GRID_SIZE", 16),
        "agent": current_agent,
        "coins": current_coins,
        "bullets": current_bullets,
    }
    next_hidden = {
        "numBullets": int(num_bullets),
    }
    return new_state, next_hidden
