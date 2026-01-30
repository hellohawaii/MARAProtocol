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


def _within_bounds(cells: List[Dict[str, Any]], grid_size: int) -> bool:
    for cell in cells:
        pos = _cell_pos(cell)
        if pos is None:
            return False
        x, y = pos
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return False
    return True


def _intersects(cells_a: List[Dict[str, Any]], cells_b: List[Dict[str, Any]]) -> bool:
    if not cells_a or not cells_b:
        return False
    return bool(_positions_set(cells_a) & _positions_set(cells_b))


def _collect_occupied(state: Dict[str, Any]) -> set[Tuple[int, int]]:
    occupied: set[Tuple[int, int]] = set()
    for key, value in state.items():
        if key in {"GRID_SIZE", "background"}:
            continue
        if not isinstance(value, list):
            continue
        occupied |= _positions_set(value)
    return occupied


def _is_free_except(
    new_cells: List[Dict[str, Any]],
    prev_cells: List[Dict[str, Any]],
    occupied: set[Tuple[int, int]],
    grid_size: int,
) -> bool:
    prev_positions = _positions_set(prev_cells)
    for cell in new_cells:
        pos = _cell_pos(cell)
        if pos is None:
            return False
        x, y = pos
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return False
        if pos in occupied and pos not in prev_positions:
            return False
    return True


def _move_no_collision(
    prev_cells: List[Dict[str, Any]],
    dx: int,
    dy: int,
    occupied: set[Tuple[int, int]],
    grid_size: int,
) -> List[Dict[str, Any]]:
    moved = _shift_cells(prev_cells, dx, dy)
    if _is_free_except(moved, prev_cells, occupied, grid_size):
        return moved
    return copy.deepcopy(prev_cells)


def _render_shape(origin: Tuple[int, int], offsets: List[Tuple[int, int]], color: str) -> List[Dict[str, Any]]:
    ox, oy = origin
    return [
        {"position": {"x": ox + dx, "y": oy + dy}, "color": color}
        for dx, dy in offsets
    ]


def _render_steps(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    offsets = [(-1, 0), (0, 0), (1, 0)]
    cells: List[Dict[str, Any]] = []
    for step in steps:
        origin = step.get("origin")
        if isinstance(origin, tuple):
            cells.extend(_render_shape(origin, offsets, "darkorange"))
    return cells


def _infer_steps_from_cells(cells: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    by_y: Dict[int, List[int]] = {}
    for cell in cells:
        pos = _cell_pos(cell)
        if pos is None:
            continue
        x, y = pos
        by_y.setdefault(y, []).append(x)
    origins: List[Tuple[int, int]] = []
    for y, xs in by_y.items():
        xs_sorted = sorted(xs)
        seq: List[int] = []
        for x in xs_sorted:
            if not seq or x == seq[-1] + 1:
                seq.append(x)
            else:
                if len(seq) == 3:
                    origins.append((seq[1], y))
                seq = [x]
        if len(seq) == 3:
            origins.append((seq[1], y))
    return sorted(origins, key=lambda p: (p[1], p[0]))


def _infer_enemy_origin(cells: List[Dict[str, Any]]) -> Tuple[int, int] | None:
    positions = [_cell_pos(cell) for cell in cells]
    positions = [pos for pos in positions if pos is not None]
    if not positions:
        return None
    min_x = min(pos[0] for pos in positions)
    min_y = min(pos[1] for pos in positions)
    return min_x + 1, min_y


def init_state():
    grid_size = 12
    mario_origin = (6, 11)
    steps_origins = [(1, 10), (5, 8), (9, 6)]
    coins_origins = [(1, 9), (7, 4), (9, 5)]
    enemy_origin = (5, 0)

    mario_cells = _render_shape(mario_origin, [(0, 0)], "red")
    steps_hidden = [{"origin": origin, "movingLeft": True} for origin in steps_origins]
    steps_cells = _render_steps(steps_hidden)
    coins_cells = [
        {"position": {"x": x, "y": y}, "color": "gold"} for x, y in coins_origins
    ]
    enemy_cells = _render_shape(
        enemy_origin,
        [(-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)],
        "blue",
    )

    state = {
        "GRID_SIZE": grid_size,
        "gtJ": mario_cells,
        "IRz": steps_cells,
        "52R": coins_cells,
        "dZ6": enemy_cells,
        "siW": [],
    }

    hidden_state = {
        "mario_bullets": 0,
        "IRz": steps_hidden,
        "enemy_movingLeft": True,
        "enemyLives": 1,
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 12), 12)
    if hidden_state is None:
        hidden_state = {}

    prev_mario_cells = copy.deepcopy(state.get("gtJ", []))
    prev_steps_cells = copy.deepcopy(state.get("IRz", []))
    prev_coins_cells = copy.deepcopy(state.get("52R", []))
    prev_enemy_cells = copy.deepcopy(state.get("dZ6", []))
    prev_bullets_cells = copy.deepcopy(state.get("siW", []))

    prev_mario_origin = _cell_pos(prev_mario_cells[0]) if prev_mario_cells else None
    prev_enemy_origin = _infer_enemy_origin(prev_enemy_cells)
    prev_enemy_defined = bool(prev_enemy_origin) and bool(prev_enemy_cells)

    inferred_step_origins = _infer_steps_from_cells(prev_steps_cells)
    hidden_steps = hidden_state.get("IRz")
    steps_hidden: List[Dict[str, Any]] = []
    if isinstance(hidden_steps, list) and len(hidden_steps) == len(inferred_step_origins):
        hidden_sorted = sorted(
            hidden_steps,
            key=lambda s: (s.get("origin", (0, 0))[1], s.get("origin", (0, 0))[0]),
        )
        for origin, step in zip(inferred_step_origins, hidden_sorted):
            steps_hidden.append({
                "origin": origin,
                "movingLeft": bool(step.get("movingLeft", True)),
            })
    else:
        steps_hidden = [{"origin": origin, "movingLeft": True} for origin in inferred_step_origins]

    prev_mario_bullets = int(hidden_state.get("mario_bullets", 0))
    prev_enemy_moving_left = bool(hidden_state.get("enemy_movingLeft", True))
    prev_enemy_lives = int(hidden_state.get("enemyLives", 1))

    current_mario_cells = copy.deepcopy(prev_mario_cells)
    current_steps_hidden = copy.deepcopy(steps_hidden)
    current_coins_cells = copy.deepcopy(prev_coins_cells)
    current_enemy_cells = copy.deepcopy(prev_enemy_cells)
    current_bullets_cells = copy.deepcopy(prev_bullets_cells)
    current_mario_bullets = prev_mario_bullets
    current_enemy_moving_left = prev_enemy_moving_left
    current_enemy_lives = prev_enemy_lives

    updated_vars: set[str] = set()
    prev_occupied = _collect_occupied(state)
    action_type = _action_type(action)

    # --- enemy boundary updates ---
    if prev_enemy_defined and prev_enemy_origin[0] == 1:
        current_enemy_cells = _shift_cells(prev_enemy_cells, 1, 0)
        current_enemy_moving_left = False
        updated_vars.add("dZ6")

    if prev_enemy_defined and prev_enemy_origin[0] == 10:
        current_enemy_cells = _shift_cells(prev_enemy_cells, -1, 0)
        current_enemy_moving_left = True
        updated_vars.add("dZ6")

    # --- mario left/right/up ---
    if action_type == "left":
        moved = _shift_cells(prev_mario_cells, -1, 0)
        if _intersects(moved, prev_coins_cells):
            current_mario_cells = moved
        else:
            current_mario_cells = _move_no_collision(
                prev_mario_cells, -1, 0, prev_occupied, grid_size
            )
        updated_vars.add("gtJ")

    if action_type == "right":
        moved = _shift_cells(prev_mario_cells, 1, 0)
        if _intersects(moved, prev_coins_cells):
            current_mario_cells = moved
        else:
            current_mario_cells = _move_no_collision(
                prev_mario_cells, 1, 0, prev_occupied, grid_size
            )
        updated_vars.add("gtJ")

    if action_type == "up":
        can_fall = _move_no_collision(prev_mario_cells, 0, 1, prev_occupied, grid_size)
        if _positions_set(can_fall) == _positions_set(prev_mario_cells):
            current_mario_cells = _move_no_collision(
                prev_mario_cells, 0, -4, prev_occupied, grid_size
            )
            updated_vars.add("gtJ")

    # --- steps movingLeft flags (on true) ---
    current_steps_hidden = []
    for step in steps_hidden:
        origin = step.get("origin")
        moving_left = bool(step.get("movingLeft", True))
        if isinstance(origin, tuple) and origin[0] == 1:
            moving_left = False
        current_steps_hidden.append({"origin": origin, "movingLeft": moving_left})
    updated_vars.add("IRz")

    current_steps_hidden = []
    for step in steps_hidden:
        origin = step.get("origin")
        moving_left = bool(step.get("movingLeft", True))
        if isinstance(origin, tuple) and origin[0] == 10:
            moving_left = True
        current_steps_hidden.append({"origin": origin, "movingLeft": moving_left})
    updated_vars.add("IRz")

    # --- mario collects coin ---
    if _intersects(prev_mario_cells, prev_coins_cells):
        mario_positions = _positions_set(prev_mario_cells)
        current_coins_cells = [
            coin for coin in prev_coins_cells if _cell_pos(coin) not in mario_positions
        ]
        current_mario_cells = _move_no_collision(
            prev_mario_cells, 0, 1, prev_occupied, grid_size
        )
        current_mario_bullets = prev_mario_bullets + 1
        updated_vars.update({"52R", "gtJ"})

    # --- click shoots bullet ---
    if action_type == "click" and prev_mario_bullets > 0 and prev_mario_origin is not None:
        current_bullets_cells = copy.deepcopy(prev_bullets_cells)
        current_bullets_cells.append(
            {"position": {"x": prev_mario_origin[0], "y": prev_mario_origin[1]}, "color": "mediumpurple"}
        )
        current_mario_cells = _move_no_collision(
            prev_mario_cells, 0, 1, prev_occupied, grid_size
        )
        current_mario_bullets = prev_mario_bullets - 1
        updated_vars.update({"siW", "gtJ"})

    # --- enemy hit by bullets ---
    if prev_enemy_defined and _intersects(prev_enemy_cells, prev_bullets_cells):
        enemy_positions = _positions_set(prev_enemy_cells)
        current_bullets_cells = [
            bullet for bullet in prev_bullets_cells if _cell_pos(bullet) not in enemy_positions
        ]
        if prev_enemy_lives == 1:
            current_enemy_cells = []
        else:
            dx = -1 if prev_enemy_moving_left else 1
            current_enemy_cells = _shift_cells(prev_enemy_cells, dx, 0)
        current_enemy_moving_left = prev_enemy_moving_left
        current_enemy_lives = prev_enemy_lives - 1
        updated_vars.update({"siW", "dZ6", "enemyLives"})

    # --- next updates (initnext) ---
    if "gtJ" not in updated_vars:
        moved_down = _shift_cells(prev_mario_cells, 0, 1)
        if _intersects(moved_down, prev_coins_cells):
            current_mario_cells = moved_down
        else:
            current_mario_cells = _move_no_collision(
                prev_mario_cells, 0, 1, prev_occupied, grid_size
            )

    if "IRz" not in updated_vars:
        current_steps_hidden = []
        for step in steps_hidden:
            origin = step.get("origin")
            moving_left = bool(step.get("movingLeft", True))
            if isinstance(origin, tuple):
                dx = -1 if moving_left else 1
                origin = (origin[0] + dx, origin[1])
            current_steps_hidden.append({"origin": origin, "movingLeft": moving_left})

    if "52R" not in updated_vars:
        current_coins_cells = copy.deepcopy(prev_coins_cells)

    if "dZ6" not in updated_vars:
        if prev_enemy_defined:
            dx = -1 if prev_enemy_moving_left else 1
            current_enemy_cells = _shift_cells(prev_enemy_cells, dx, 0)
        else:
            current_enemy_cells = []

    if "siW" not in updated_vars:
        current_bullets_cells = []
        for bullet in prev_bullets_cells:
            moved = _shift_cells([bullet], 0, -1)[0]
            if _intersects([moved], prev_steps_cells):
                current_bullets_cells.append(copy.deepcopy(bullet))
            else:
                current_bullets_cells.append(copy.deepcopy(moved))

    if "enemyLives" not in updated_vars:
        current_enemy_lives = prev_enemy_lives

    next_state = {
        "GRID_SIZE": grid_size,
        "gtJ": current_mario_cells,
        "IRz": _render_steps(current_steps_hidden),
        "52R": current_coins_cells,
        "dZ6": current_enemy_cells,
        "siW": current_bullets_cells,
    }

    next_hidden = {
        "mario_bullets": int(current_mario_bullets),
        "IRz": current_steps_hidden,
        "enemy_movingLeft": bool(current_enemy_moving_left),
        "enemyLives": int(current_enemy_lives),
    }
    return next_state, next_hidden
