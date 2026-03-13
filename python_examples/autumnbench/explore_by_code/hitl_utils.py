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

    init_state_func, predict_dynamics_func, compile_error = _compile_functions(code_str)

    results = []

    for idx, traj_data in enumerate(trajectories):
        traj_name = traj_data.get("__traj_name", f"traj_{idx}")
        transitions = traj_data.get("trajectory", [])
        
        if not transitions:
            continue
            
        framesGT = []
        framesPred = []
        frameActions = []
        
        # Add initial state
        initial_state = transitions[0]["state"]
        framesGT.append({"rawFrame": initial_state})
        framesPred.append({"rawFrame": initial_state})

        # Pre-populate GT and Actions
        for i, transition in enumerate(transitions):
            action_str = transition["action"]
            action_obj = {"type": "noop"}
            if action_str in ["left", "right", "up", "down", "noop"]:
                action_obj = {"type": action_str}
            elif action_str.startswith("click"):
                parts = action_str.split()
                if len(parts) >= 3:
                    action_obj = {"type": "click", "x": int(parts[1]), "y": int(parts[2])}
            frameActions.append(action_obj)
            framesGT.append({"rawFrame": transition["new_state"]})

        correct_predictions = 0
        total_predictions = len(transitions)
        
        if compile_error:
            for _ in transitions:
                framesPred.append({"rawFrame": None, "error": compile_error, "is_correct": False})
            accuracy = 0
        else:
            _, hidden_state, error = _call_init_state(init_state_func)
            if error:
                for _ in transitions:
                    framesPred.append({"rawFrame": None, "error": error, "is_correct": False})
                accuracy = 0
            else:
                last_predicted_visible_state = None
                for i, transition in enumerate(transitions):
                    if i == 0:
                        state = transition["state"]
                    else:
                        state = last_predicted_visible_state

                    action_str = transition["action"]
                    
                    next_state, hidden_state, error = _call_predict(
                        predict_dynamics_func,
                        state,
                        hidden_state,
                        action_str,
                    )
                    
                    if error:
                        framesPred.append({"rawFrame": None, "error": error, "is_correct": False})
                        # Fill remaining with error
                        for _ in range(i + 1, len(transitions)):
                            framesPred.append({"rawFrame": None, "error": "Previous step failed", "is_correct": False})
                        break

                    last_predicted_visible_state = next_state
                    ground_truth_next_state = transition["new_state"]
                    
                    is_correct = compare_visible_states(next_state, ground_truth_next_state)
                    framesPred.append({"rawFrame": next_state, "is_correct": is_correct})
                    
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
        "results": results,
        "compile_error": compile_error
    }

