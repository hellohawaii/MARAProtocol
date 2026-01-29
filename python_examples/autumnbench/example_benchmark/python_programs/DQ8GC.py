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


def _particle_health(cell: Dict[str, Any]) -> bool:
    if "health" in cell:
        return bool(cell.get("health"))
    return cell.get("color") == "gray"


def _particle_color(health: bool) -> str:
    return "gray" if health else "darkgreen"


def _set_particle_health(cell: Dict[str, Any], health: bool) -> Dict[str, Any]:
    cell["health"] = bool(health)
    cell["color"] = _particle_color(bool(health))
    return cell


def _make_particle(x: int, y: int, health: bool) -> Dict[str, Any]:
    return {
        "position": {"x": int(x), "y": int(y)},
        "color": _particle_color(health),
        "health": bool(health),
    }


def _adjacent(pos1: Tuple[int, int], pos2: Tuple[int, int], unit: int = 1) -> bool:
    return abs(pos2[0] - pos1[0]) + abs(pos2[1] - pos1[1]) <= unit


def _move_particle(cell: Dict[str, Any], dx: int, dy: int) -> Dict[str, Any]:
    moved = copy.deepcopy(cell)
    pos = moved.get("position", {})
    if isinstance(pos, dict):
        pos["x"] = _safe_int(pos.get("x")) + dx
        pos["y"] = _safe_int(pos.get("y")) + dy
    return moved


def _clicked_pos(action: Any) -> Tuple[int, int] | None:
    if isinstance(action, dict) and action.get("type") == "click":
        return _safe_int(action.get("x")), _safe_int(action.get("y"))
    return None


def init_state():
    grid_size = 16
    inactive = [
        _make_particle(7, 5, True),
        _make_particle(4, 3, True),
        _make_particle(6, 6, True),
        _make_particle(3, 5, True),
    ]
    active = _make_particle(2, 2, False)
    state = {
        "GRID_SIZE": grid_size,
        "inactiveParticles": inactive,
        "activeParticle": [active],
    }
    return state, {}


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    new_state = copy.deepcopy(state)
    prev_inactive = copy.deepcopy(state.get("inactiveParticles", []))
    prev_active_list = copy.deepcopy(state.get("activeParticle", []))
    prev_active = prev_active_list[0] if prev_active_list else None

    action_type = _action_type(action)
    updated_vars: set[str] = set()

    current_active = copy.deepcopy(prev_active) if prev_active is not None else None

    # --- on adjacent unhealthy ---
    if prev_active is not None:
        active_pos = _cell_pos(prev_active)
        if active_pos is not None:
            any_unhealthy = False
            for obj in [prev_active] + prev_inactive:
                if not _particle_health(obj):
                    obj_pos = _cell_pos(obj)
                    if obj_pos is not None and _adjacent(active_pos, obj_pos, 1):
                        any_unhealthy = True
                        break
            if any_unhealthy:
                current_active = _set_particle_health(copy.deepcopy(prev_active), False)
                new_state["activeParticle"] = [current_active]
                updated_vars.add("activeParticle")

    # --- on clicked swap ---
    click_pos = _clicked_pos(action)
    if click_pos is not None:
        clicked_particle = None
        for obj in prev_inactive:
            if _cell_pos(obj) == click_pos:
                clicked_particle = copy.deepcopy(obj)
                break
        if clicked_particle is not None:
            remaining = []
            used = False
            for obj in prev_inactive:
                if not used and _cell_pos(obj) == click_pos:
                    used = True
                    continue
                remaining.append(copy.deepcopy(obj))
            if current_active is not None:
                remaining.append(copy.deepcopy(current_active))
            new_state["inactiveParticles"] = remaining
            new_state["activeParticle"] = [clicked_particle]
            current_active = clicked_particle
            updated_vars.update({"inactiveParticles", "activeParticle"})

    # --- movement on arrows (uses prev activeParticle) ---
    if prev_active is not None and action_type in {"left", "right", "up", "down"}:
        dx = 0
        dy = 0
        if action_type == "left":
            dx = -1
        elif action_type == "right":
            dx = 1
        elif action_type == "up":
            dy = -1
        elif action_type == "down":
            dy = 1
        moved = _move_particle(prev_active, dx, dy)
        new_state["activeParticle"] = [moved]
        current_active = moved
        updated_vars.add("activeParticle")

    # --- initnext update for inactiveParticles ---
    if "inactiveParticles" not in updated_vars:
        unhealthy_sources = []
        if prev_active is not None and not _particle_health(prev_active):
            unhealthy_sources.append(prev_active)
        for obj in prev_inactive:
            if not _particle_health(obj):
                unhealthy_sources.append(obj)

        next_inactive = []
        for obj in prev_inactive:
            updated_obj = copy.deepcopy(obj)
            pos = _cell_pos(obj)
            if pos is not None and unhealthy_sources:
                for source in unhealthy_sources:
                    src_pos = _cell_pos(source)
                    if src_pos is not None and _adjacent(pos, src_pos, 1):
                        updated_obj = _set_particle_health(updated_obj, False)
                        break
            next_inactive.append(updated_obj)
        new_state["inactiveParticles"] = next_inactive

    # Normalize colors for visibility.
    if "inactiveParticles" in new_state and isinstance(new_state["inactiveParticles"], list):
        for obj in new_state["inactiveParticles"]:
            _set_particle_health(obj, _particle_health(obj))
    if "activeParticle" in new_state and isinstance(new_state["activeParticle"], list):
        for obj in new_state["activeParticle"]:
            _set_particle_health(obj, _particle_health(obj))

    return new_state, hidden_state
