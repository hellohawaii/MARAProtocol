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


def _clicked_pos(action: Any) -> Tuple[int, int] | None:
    if isinstance(action, dict) and action.get("type") == "click":
        return _safe_int(action.get("x")), _safe_int(action.get("y"))
    return None


def init_state():
    grid_size = 16
    state = {
        "GRID_SIZE": grid_size,
        "particles": [],
    }
    hidden_state = {
        "currColor": "red",
        "active_arrow": "none",
    }
    return state, hidden_state


def predict_dynamics(state: Dict[str, Any], hidden_state: Dict[str, Any] | None, action: Any):
    if not isinstance(state, dict):
        return state, hidden_state

    grid_size = _safe_int(state.get("GRID_SIZE", 16), 16)
    prev_particles = copy.deepcopy(state.get("particles", []))

    if hidden_state is None:
        _, hidden_state = init_state()

    curr_color = str(hidden_state.get("currColor", "red"))
    active_arrow = str(hidden_state.get("active_arrow", "none"))
    prev_active_arrow = active_arrow

    action_type = _action_type(action)

    # --- on clicked ---
    click_pos = _clicked_pos(action)
    if click_pos is not None:
        particle_color = "red" if active_arrow == "none" else curr_color
        prev_particles.append({
            "position": {"x": click_pos[0], "y": click_pos[1]},
            "color": particle_color,
        })

    # --- arrow updates ---
    if action_type == "up":
        curr_color = "gold"
        active_arrow = "up"
    elif action_type == "down":
        curr_color = "purple"
        active_arrow = "down"
    elif action_type == "left":
        curr_color = "green"
        active_arrow = "left"
    elif action_type == "right":
        curr_color = "blue"
        active_arrow = "right"

    # --- reset opposing arrows (uses prev active_arrow) ---
    if action_type == "down" and prev_active_arrow == "up":
        curr_color = "red"
        active_arrow = "none"
    if action_type == "up" and prev_active_arrow == "down":
        curr_color = "red"
        active_arrow = "none"
    if action_type == "left" and prev_active_arrow == "right":
        curr_color = "red"
        active_arrow = "none"
    if action_type == "right" and prev_active_arrow == "left":
        curr_color = "red"
        active_arrow = "none"

    new_state = {
        "GRID_SIZE": grid_size,
        "particles": prev_particles,
    }
    next_hidden = {
        "currColor": curr_color,
        "active_arrow": active_arrow,
    }
    return new_state, next_hidden
