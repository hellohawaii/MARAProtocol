import argparse
import copy
import contextlib
import io
import json
import os
import signal
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

# --- Sandboxed Code Execution Utilities ---

class TimeoutException(Exception):
    pass

class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""
    def read(self, *args, **kwargs):
        raise OSError
    def readline(self, *args, **kwargs):
        raise OSError
    def readlines(self, *args, **kwargs):
        raise OSError
    def readable(self, *args, **kwargs):
        return False

class redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"

@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    # Check if we are in the main thread
    import threading
    if threading.current_thread() is not threading.main_thread():
        # We are in a background thread. signal.setitimer only works in main thread.
        # But we can't easily interrupt a thread in Python.
        # For now, we will just yield, but ideally we should use multiprocessing.
        # Actually, let's use a trace function to implement timeout in thread.
        import sys
        import time
        start_time = time.time()
        def trace_calls(frame, event, arg):
            if time.time() - start_time > seconds:
                raise TimeoutException("Timed out!")
            return trace_calls
        
        old_trace = sys.gettrace()
        sys.settrace(trace_calls)
        try:
            yield
        finally:
            sys.settrace(old_trace)
        return

    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        signal.signal(signal.SIGALRM, signal_handler)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield stream

def eval_code(line, timeout=3., return_exec_globals=False, exec_globals=None):
    try:
        exec_globals = {} if exec_globals is None else exec_globals
        with swallow_io() as s:
            with time_limit(timeout):
                exec(line, exec_globals)
        if return_exec_globals:
            return exec_globals
        else:
            return 'passed'
    except TimeoutException:
        return 'timed out'
    except BaseException as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        extracted_tb = traceback.extract_tb(exc_traceback)
        exec_tb_frames = [frame for frame in extracted_tb if frame.filename == '<string>']

        if exec_tb_frames:
            exec_stack_trace_list = traceback.format_list(exec_tb_frames)
            exception_only_list = traceback.format_exception_only(exc_type, exc_value)
            exc_str = "Traceback (most recent call last):\n" + "".join(exec_stack_trace_list) + "".join(exception_only_list)
        else:
            exception_only_list = traceback.format_exception_only(exc_type, exc_value)
            exc_str = "".join(exception_only_list)

        try:
            output = s.getvalue()
            str(output)
        except Exception as ee:
            return f'failed: {exc_str}\nfailed: in printing outputs: {ee}'
        try:
            str(e)
        except Exception as ee:
            return f'failed: in printing exception: {ee}\nPrinted outputs: {output}'
        return f"failed: {exc_str}\nPrinted outputs: {output}"


def state_to_grid(state: Dict[str, Any]) -> Optional[List[List[str]]]:
    """Converts a raw visible state dictionary to a 2D grid of color strings."""
    if not state or "GRID_SIZE" not in state:
        return None

    grid_size = state["GRID_SIZE"]
    grid = [["black" for _ in range(grid_size)] for _ in range(grid_size)]

    object_types = sorted(state.keys())
    if "background" in object_types:
        object_types.insert(0, object_types.pop(object_types.index("background")))

    for obj_type in object_types:
        if obj_type == "GRID_SIZE":
            continue
        
        objects = state.get(obj_type)
        if not isinstance(objects, list):
            continue

        for obj in objects:
            if isinstance(obj, dict) and "position" in obj and "color" in obj:
                pos = obj.get("position", {})
                if isinstance(pos, dict) and "x" in pos and "y" in pos:
                    x, y, color = pos["x"], pos["y"], obj["color"]
                    if 0 <= y < grid_size and 0 <= x < grid_size:
                        grid[y][x] = color
    return grid


def compare_visible_states(state1, state2):
    """
    Compares two visible state dictionaries by converting them to grids and
    checking for grid equality.
    """
    if state1 is None and state2 is None:
        return True
    if state1 is None or state2 is None:
        return False
        
    grid1 = state_to_grid(state1)
    grid2 = state_to_grid(state2)

    if grid1 is None or grid2 is None:
        return grid1 == grid2

    return grid1 == grid2

# --- Core Evaluation Logic ---

def read_code(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Code file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _compile_functions(
    code_str: str,
) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    exec_globals = eval_code(code_str, return_exec_globals=True)
    if isinstance(exec_globals, str):
        return None, None, exec_globals

    init_state_func = exec_globals.get("init_state")
    predict_dynamics_func = exec_globals.get("predict_dynamics")

    if not callable(init_state_func):
        return None, None, "Function 'init_state' not found or not callable."
    if not callable(predict_dynamics_func):
        return None, None, "Function 'predict_dynamics' not found or not callable."

    return init_state_func, predict_dynamics_func, None


def _call_init_state(init_state_func: Any) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    call_globals = {"init_state_func": init_state_func}
    result_or_error = eval_code(
        "__result__ = init_state_func()",
        exec_globals=call_globals,
        return_exec_globals=True,
    )
    if isinstance(result_or_error, str):
        return None, None, result_or_error
    result = result_or_error.get("__result__")
    if not isinstance(result, tuple) or len(result) != 2:
        return None, None, "init_state() must return (state, hidden_state)."
    return result[0], result[1], None


def _call_predict(
    predict_dynamics_func: Any,
    state: Any,
    hidden_state: Any,
    action: Any,
) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    call_globals = {
        "predict_dynamics_func": predict_dynamics_func,
        "state": copy.deepcopy(state),
        "hidden_state": copy.deepcopy(hidden_state),
        "action": action,
    }
    result_or_error = eval_code(
        "__result__ = predict_dynamics_func(state, hidden_state, action)",
        exec_globals=call_globals,
        return_exec_globals=True,
    )
    if isinstance(result_or_error, str):
        return None, None, result_or_error
    result = result_or_error.get("__result__")
    if not isinstance(result, tuple) or len(result) != 2:
        return None, None, "predict_dynamics() must return (next_state, next_hidden_state)."
    return result[0], result[1], None


def load_trajectories(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory path not found: {path}")

    trajectories = []
    
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in sorted(files):
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "trajectory" in data:
                            data["__traj_name"] = os.path.relpath(file_path, path)
                            trajectories.append(data)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "trajectory" in data:
                data["__traj_name"] = os.path.basename(path)
                trajectories.append(data)
                
    return trajectories


def evaluate(code_str: str, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trajectories:
        return {"success": False, "error": "No trajectories loaded."}

    init_state_func, predict_dynamics_func, error = _compile_functions(code_str)
    if error:
        return {"success": False, "error": error}

    total_steps = 0
    total_correct = 0
    per_traj_accuracy = []
    per_trajectory_stats = []

    for idx, traj_data in enumerate(trajectories):
        traj_name = traj_data.get("__traj_name", f"traj_{idx}")
        transitions = traj_data.get("trajectory", [])
        if not transitions:
            per_traj_accuracy.append(0)
            per_trajectory_stats.append(
                {
                    "traj_name": traj_name,
                    "correct_frames": 0,
                    "total_frames": 0,
                    "accuracy": 0,
                }
            )
            continue
            
        _, hidden_state, error = _call_init_state(init_state_func)
        if error:
            return {"success": False, "error": f"Trajectory {idx}: " + error}

        correct_predictions = 0
        total_predictions = len(transitions)
        
        last_predicted_visible_state = None

        for i, transition in enumerate(transitions):
            if i == 0:
                state = transition["state"]
            else:
                state = last_predicted_visible_state

            action_info = transition["action"]

            next_state, hidden_state, error = _call_predict(
                predict_dynamics_func,
                state,
                hidden_state,
                action_info,
            )
            if error:
                return {"success": False, "error": f"Trajectory {idx} step {i}: " + error}

            last_predicted_visible_state = next_state
            ground_truth_next_state = transition["new_state"]
            
            is_correct = compare_visible_states(next_state, ground_truth_next_state)
            if is_correct:
                correct_predictions += 1

        accuracy = (correct_predictions / total_predictions) if total_predictions > 0 else 0
        per_traj_accuracy.append(accuracy)
        per_trajectory_stats.append(
            {
                "traj_name": traj_name,
                "correct_frames": correct_predictions,
                "total_frames": total_predictions,
                "accuracy": accuracy,
            }
        )
        total_steps += total_predictions
        total_correct += correct_predictions

    overall_accuracy = (total_correct / total_steps) if total_steps > 0 else 0.0
    is_perfect = total_steps > 0 and total_correct == total_steps

    return {
        "success": True,
        "overall_accuracy": overall_accuracy,
        "total_steps": total_steps,
        "total_correct": total_correct,
        "is_perfect": is_perfect,
        "per_trajectory_accuracy": per_traj_accuracy,
        "per_trajectory_stats": per_trajectory_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check python code accuracy on JSON trajectories with rollout."
    )
    parser.add_argument(
        "code_path",
        help="Python file defining init_state() and predict_dynamics().",
    )
    parser.add_argument(
        "trajectory_path",
        help="Path to trajectory .json file or a directory containing JSON trajectories.",
    )
    args = parser.parse_args()

    try:
        code_str = read_code(args.code_path)
    except FileNotFoundError as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1

    try:
        trajectories = load_trajectories(args.trajectory_path)
    except FileNotFoundError as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1

    result = evaluate(code_str, trajectories)
    print(json.dumps(result, indent=2, ensure_ascii=True))

    if not result.get("success"):
        return 1
    return 0 if result.get("is_perfect") else 2

if __name__ == "__main__":
    raise SystemExit(main())
