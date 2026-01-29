import copy
from typing import Any, Dict, Tuple


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _action_type(action: Any) -> str:
    if isinstance(action, dict):
        return str(action.get("type", "noop"))
    return str(action) if action is not None else "noop"


def _pos(cell: Dict[str, Any]) -> Tuple[int, int] | None:
    pos = cell.get("position")
    if isinstance(pos, dict) and "x" in pos and "y" in pos:
        return _safe_int(pos["x"]), _safe_int(pos["y"])
    return None


def _build_pos_index(state: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for value in state.values():
        if not isinstance(value, list):
            continue
        for cell in value:
            if not isinstance(cell, dict):
                continue
            pos = _pos(cell)
            if pos is not None:
                index[pos] = cell
    return index


def _infer_health(pos_to_color: Dict[Tuple[int, int], str], grid_size: int) -> int:
    max_blue = None
    for x in range(grid_size):
        if pos_to_color.get((x, 0)) == "blue":
            max_blue = x
    return -1 if max_blue is None else max_blue


def _infer_fire(pos_to_color: Dict[Tuple[int, int], str], fire_pos: Tuple[int, int]) -> bool:
    return pos_to_color.get(fire_pos) == "orange"


def _infer_gas(pos_to_color: Dict[Tuple[int, int], str],
               gas_hi_pos: Tuple[int, int],
               gas_lo_pos: Tuple[int, int]) -> int:
    gas_hi = pos_to_color.get(gas_hi_pos)
    gas_lo = pos_to_color.get(gas_lo_pos)
    if gas_lo != "yellow":
        return 0
    if gas_hi == "yellow":
        return 65
    return 20


def _infer_cooked(pos_to_color: Dict[Tuple[int, int], str], meat_pos: Tuple[int, int]) -> int:
    color = pos_to_color.get(meat_pos)
    if color == "lightblue":
        return 0
    if color == "pink":
        return 10
    if color == "sandybrown":
        return 30
    if color == "brown":
        return 60
    return 0


def _cooked_color(cooked: int) -> str:
    if cooked < 10:
        return "lightblue"
    if cooked < 30:
        return "pink"
    if cooked < 60:
        return "sandybrown"
    return "brown"


def _bbq_positions(grid_size: int) -> Tuple[Tuple[int, int], ...]:
    origin_x = grid_size // 2 - 1
    origin_y = grid_size - 4
    rels = [
        (0, 1), (2, 1), (1, 1),
        (0, 2), (0, 3),
        (2, 2), (2, 3),
        (1, 2), (1, 3),
    ]
    return tuple((origin_x + dx, origin_y + dy) for dx, dy in rels)


def init_state():
    grid_size = 7
    origin_x = grid_size // 2 - 1
    origin_y = grid_size - 4
    fire = True
    gas = 65
    cooked = 0
    health = grid_size // 2

    fire_pos = (origin_x + 1, origin_y + 1)
    gas_hi_pos = (origin_x + 1, origin_y + 2)
    gas_lo_pos = (origin_x + 1, origin_y + 3)

    bbq_cells = []
    for pos in _bbq_positions(grid_size):
        color = "gray"
        if pos == fire_pos:
            color = "orange" if fire else "white"
        elif pos == gas_hi_pos:
            color = "yellow" if gas > 20 else "white"
        elif pos == gas_lo_pos:
            color = "yellow" if gas > 0 else "white"
        bbq_cells.append({"position": {"x": pos[0], "y": pos[1]}, "color": color})

    meat_pos = (grid_size // 2, grid_size - 4)
    meat_cells = [{"position": {"x": meat_pos[0], "y": meat_pos[1]}, "color": _cooked_color(cooked)}]

    fill_pos = (0, grid_size - 1)
    fill_cells = [{"position": {"x": fill_pos[0], "y": fill_pos[1]}, "color": "yellow"}]

    person_cells = []
    for x in range(grid_size):
        color = "blue" if x <= health else "black"
        person_cells.append({"position": {"x": x, "y": 0}, "color": color})

    state = {
        "GRID_SIZE": grid_size,
        "bbq": bbq_cells,
        "meat": meat_cells,
        "fillButton": fill_cells,
        "person": person_cells,
    }

    hidden_state = {
        "gas": gas,
        "cooked": cooked,
        "fire": fire,
        "health": health,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    new_state = copy.deepcopy(state)
    grid_size = _safe_int(state.get("GRID_SIZE", 7), 7)

    origin_x = grid_size // 2 - 1
    origin_y = grid_size - 4
    fire_pos = (origin_x + 1, origin_y + 1)
    gas_hi_pos = (origin_x + 1, origin_y + 2)
    gas_lo_pos = (origin_x + 1, origin_y + 3)
    meat_pos = (grid_size // 2, grid_size - 4)
    fill_pos = (0, grid_size - 1)
    bbq_cells = set(_bbq_positions(grid_size))

    pos_index = _build_pos_index(new_state)
    pos_to_color = {pos: cell.get("color") for pos, cell in pos_index.items()}

    inferred_gas = _infer_gas(pos_to_color, gas_hi_pos, gas_lo_pos)
    inferred_cooked = _infer_cooked(pos_to_color, meat_pos)
    inferred_fire = _infer_fire(pos_to_color, fire_pos)
    inferred_health = _infer_health(pos_to_color, grid_size)

    if hidden_state is None:
        _, hidden_state = init_state()
        gas = int(hidden_state["gas"])
        cooked = int(hidden_state["cooked"])
        fire = bool(hidden_state["fire"])
        health = int(hidden_state["health"])
    else:
        gas = int(hidden_state.get("gas", inferred_gas))
        cooked = int(hidden_state.get("cooked", inferred_cooked))
        fire = bool(hidden_state.get("fire", inferred_fire))
        health = int(hidden_state.get("health", inferred_health))

    prev_gas = gas
    prev_fire = fire
    prev_cooked = cooked
    prev_health = health

    # --- initnext updates ---
    next_fire = prev_fire
    next_gas = prev_gas
    if prev_gas == 0:
        next_fire = False
    elif prev_fire:
        next_gas = max(0, prev_gas - 1)

    next_health = prev_health

    # --- action updates ---
    action_type = _action_type(action)
    if action_type == "click" and isinstance(action, dict):
        ax = _safe_int(action.get("x"))
        ay = _safe_int(action.get("y"))
        click_pos = (ax, ay)

        if click_pos in bbq_cells:
            if prev_gas != 0:
                next_fire = not prev_fire
            else:
                next_fire = prev_fire
            # on-clause update overrides next; keep gas unchanged this step
            next_gas = prev_gas

        if click_pos == fill_pos:
            next_gas = prev_gas + 5

    if action_type == "click" and isinstance(action, dict):
        if click_pos == meat_pos:
            if next_health >= 0:
                if prev_cooked < 30:
                    delta = -1
                elif prev_cooked > 60:
                    delta = -2
                else:
                    delta = 1
                next_health = max(-1, min(grid_size - 1, next_health + delta))
            else:
                next_health = prev_health
            next_cooked = 0
        else:
            # Meat cooks after BBQ updates (including click toggles).
            next_cooked = prev_cooked + (1 if next_fire else 0)
    else:
        # Meat cooks after BBQ updates (including click toggles).
        next_cooked = prev_cooked + (1 if next_fire else 0)

    # --- render updates ---
    def set_color(pos: Tuple[int, int], color: str):
        cell = pos_index.get(pos)
        if cell is not None:
            cell["color"] = color

    # Person row
    for x in range(grid_size):
        color = "blue" if x <= next_health else "black"
        set_color((x, 0), color)

    # BBQ frame
    for pos in bbq_cells:
        set_color(pos, "gray")
    set_color(fire_pos, "orange" if next_fire else "white")
    set_color(gas_hi_pos, "yellow" if next_gas > 20 else "white")
    set_color(gas_lo_pos, "yellow" if next_gas > 0 else "white")

    # Meat and fill button
    set_color(meat_pos, _cooked_color(next_cooked))
    set_color(fill_pos, "yellow")

    next_hidden = {
        "gas": int(next_gas),
        "cooked": int(next_cooked),
        "fire": bool(next_fire),
        "health": int(next_health),
    }

    return new_state, next_hidden
