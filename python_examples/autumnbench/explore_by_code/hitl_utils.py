import json
import os
import copy
from pathlib import Path
from typing import Dict, Any, List

# Import the evaluation logic from check_traj_example
from llm_workspace.check_traj_example import _compile_functions, _call_init_state, _call_predict, compare_visible_states

def evaluate_model_on_trajectories(code_str: str, traj_dir: str) -> Dict[str, Any]:
    """
    Evaluates the code on all trajectories in traj_dir.
    Returns a dictionary suitable for sending to the frontend.
    """
    # Load trajectories
    trajectories = []
    traj_path = Path(traj_dir)
    if traj_path.exists() and traj_path.is_dir():
        for file_path in sorted(traj_path.glob("*.json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "trajectory" in data:
                        data["__traj_name"] = file_path.name
                        trajectories.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    if not trajectories:
        return {"success": False, "error": "No trajectories found."}

    init_state_func, predict_dynamics_func, error = _compile_functions(code_str)
    if error:
        return {"success": False, "error": error}

    results = []

    for idx, traj_data in enumerate(trajectories):
        traj_name = traj_data.get("__traj_name", f"traj_{idx}")
        transitions = traj_data.get("trajectory", [])
        
        if not transitions:
            continue
            
        _, hidden_state, error = _call_init_state(init_state_func)
        if error:
            return {"success": False, "error": f"Trajectory {traj_name}: " + error}

        correct_predictions = 0
        total_predictions = len(transitions)
        
        last_predicted_visible_state = None
        
        # We need to build framesGT and framesPred for the frontend
        # framesGT: [initial_state, new_state_1, new_state_2, ...]
        # framesPred: [initial_state, pred_state_1, pred_state_2, ...]
        # frameActions: [action_1, action_2, ...]
        
        framesGT = []
        framesPred = []
        frameActions = []
        
        # Add initial state
        initial_state = transitions[0]["state"]
        framesGT.append({"rawFrame": initial_state})
        framesPred.append({"rawFrame": initial_state})

        for i, transition in enumerate(transitions):
            if i == 0:
                state = transition["state"]
            else:
                state = last_predicted_visible_state

            action_str = transition["action"]
            
            # Convert action_str to frontend action format if needed
            # Frontend expects: { type: 'click', x: 1, y: 2 } or { type: 'left' }
            action_obj = {"type": "noop"}
            if action_str in ["left", "right", "up", "down", "noop"]:
                action_obj = {"type": action_str}
            elif action_str.startswith("click"):
                parts = action_str.split()
                if len(parts) >= 3:
                    action_obj = {"type": "click", "x": int(parts[1]), "y": int(parts[2])}
            
            frameActions.append(action_obj)

            # The code expects action as a string (as in check_traj_example.py)
            next_state, hidden_state, error = _call_predict(
                predict_dynamics_func,
                state,
                hidden_state,
                action_str,
            )
            
            if error:
                # If error, we append the error state or just break
                framesPred.append({"rawFrame": None, "error": error})
                break

            last_predicted_visible_state = next_state
            ground_truth_next_state = transition["new_state"]
            
            framesGT.append({"rawFrame": ground_truth_next_state})
            framesPred.append({"rawFrame": next_state})
            
            is_correct = compare_visible_states(next_state, ground_truth_next_state)
            if is_correct:
                correct_predictions += 1

        accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0
        
        results.append({
            "traj_name": traj_name,
            "framesGT": framesGT,
            "framesPred": framesPred,
            "frameActions": frameActions,
            "accuracy": accuracy,
            "correct_frames": correct_predictions,
            "total_frames": total_predictions
        })

    return {
        "success": True,
        "results": results
    }

