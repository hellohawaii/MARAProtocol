"""
Code refinement subgraph for iteratively improving predict_dynamics code.

This module implements a LangGraph subgraph that:
1. Generates initial predict_dynamics code from trajectories
2. Iteratively refines code by executing it, finding first-error frames, and
   asking an LLM to fix issues based on single-frame error information
3. Generates diagnostic questions when refinement cannot reach 100% accuracy

The subgraph maintains a conversation history across iterations so the LLM can
detect regressions and track progress.
"""

import copy
import contextlib
import hashlib
import io
import json
import logging
import multiprocessing
import os
import pickle
import re
import signal
import sys
import time
import traceback
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, BaseMessage):
        return {
            "type": getattr(obj, "type", obj.__class__.__name__),
            "content": getattr(obj, "content", ""),
        }
    return obj


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


_BASIC_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 20, 60),
    "blue": (65, 105, 225),
    "green": (50, 205, 50),
    "yellow": (255, 215, 0),
    "orange": (255, 140, 0),
    "purple": (138, 43, 226),
    "pink": (255, 105, 180),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "slategrey": (112, 128, 144),
    "slategray": (112, 128, 144),
}


def _color_to_rgb(color_name: Any) -> tuple:
    if isinstance(color_name, str):
        c = color_name.strip().lower()
        if c in _BASIC_COLORS:
            return _BASIC_COLORS[c]
        if c.startswith("#") and len(c) == 7:
            try:
                return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
            except Exception:
                pass
    seed = abs(hash(str(color_name))) % (256 ** 3)
    return ((seed >> 16) & 255, (seed >> 8) & 255, seed & 255)


def _render_grid_png(grid: List[List[str]], out_path: str, cell_size: int = 24) -> None:
    if Image is None or ImageDraw is None:
        return
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return
    img = Image.new("RGB", (w * cell_size, h * cell_size), "black")
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, color_name in enumerate(row):
            rgb = _color_to_rgb(color_name)
            x0, y0 = x * cell_size, y * cell_size
            x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=rgb)
            draw.rectangle([x0, y0, x1, y1], outline=(40, 40, 40))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def _log_refine_llm_call(state: Dict[str, Any], node_name: str, llm_input: Any,
                         llm_output: Any, error: Optional[str] = None) -> None:
    log_dir = state.get("code_refine_log_dir")
    if not log_dir:
        return
    ts = int(time.time() * 1000)
    out_path = os.path.join(log_dir, "llm_calls", f"{ts}_{node_name}.json")
    _write_json(out_path, {
        "node": node_name,
        "main_loop_index": state.get("main_loop_index"),
        "refine_iteration": state.get("refine_iteration"),
        "llm_input": llm_input,
        "llm_output": llm_output,
        "error": error,
    })


def _persist_refine_eval_artifacts(
    state: Dict[str, Any],
    results_list: List[Dict[str, Any]],
    accuracy: float,
    total_correct: int,
    total_frames: int,
    prior_eval: Optional[Dict[str, Any]] = None,
) -> int:
    log_dir = state.get("code_refine_log_dir")
    curr_eval_idx = int(state.get("refine_eval_index", 0))
    if not log_dir:
        return curr_eval_idx + 1

    refine_dir = os.path.join(log_dir, f"refine_loop_{curr_eval_idx:03d}")
    os.makedirs(refine_dir, exist_ok=True)

    _write_json(os.path.join(refine_dir, "overall_metrics.json"), {
        "main_loop_index": state.get("main_loop_index"),
        "refine_loop_index": curr_eval_idx,
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_frames": total_frames,
        "num_trajectories": len(results_list),
        "prior_trajectory_eval": prior_eval or {},
    })
    _write_text(os.path.join(refine_dir, "predict_dynamics.py"), state.get("code") or "")

    for traj_idx, traj_result in enumerate(results_list):
        traj_dir = os.path.join(refine_dir, f"trajectory_{traj_idx:03d}")
        os.makedirs(traj_dir, exist_ok=True)
        _write_json(os.path.join(traj_dir, "metrics.json"), {
            "accuracy": traj_result.get("accuracy", 0.0),
            "correct": traj_result.get("correct", 0),
            "total": traj_result.get("total", 0),
        })
        _write_text(os.path.join(traj_dir, "predict_dynamics.py"), state.get("code") or "")

        first_error_step = None
        for step in traj_result.get("steps", []):
            if not step.get("is_correct", False):
                first_error_step = step
                break
        if first_error_step is None:
            continue

        _write_json(os.path.join(traj_dir, "first_error_frame.json"), {
            "step_idx": first_error_step.get("step_idx"),
            "input_visible_state": first_error_step.get("input_visible_state"),
            "action": first_error_step.get("action"),
            "predicted_visible_state": first_error_step.get("predicted_visible_state"),
            "ground_truth_next_state": first_error_step.get("ground_truth_next_state"),
            "final_hidden_state": first_error_step.get("final_hidden_state"),
        })

        pred_grid = _state_to_grid(first_error_step.get("predicted_visible_state"))
        gt_grid = _state_to_grid(first_error_step.get("ground_truth_next_state"))
        if pred_grid is not None:
            _render_grid_png(pred_grid, os.path.join(traj_dir, "predicted_next.png"))
        if gt_grid is not None:
            _render_grid_png(gt_grid, os.path.join(traj_dir, "ground_truth_next.png"))
    return curr_eval_idx + 1


def _persist_refine_eval_error(state: Dict[str, Any], error_message: str,
                               prior_eval: Optional[Dict[str, Any]] = None) -> int:
    log_dir = state.get("code_refine_log_dir")
    curr_eval_idx = int(state.get("refine_eval_index", 0))
    if not log_dir:
        return curr_eval_idx + 1
    refine_dir = os.path.join(log_dir, f"refine_loop_{curr_eval_idx:03d}")
    os.makedirs(refine_dir, exist_ok=True)
    _write_json(os.path.join(refine_dir, "execution_error.json"), {
        "main_loop_index": state.get("main_loop_index"),
        "refine_loop_index": curr_eval_idx,
        "error": error_message,
        "prior_trajectory_eval": prior_eval or {},
    })
    _write_text(os.path.join(refine_dir, "predict_dynamics.py"), state.get("code") or "")
    return curr_eval_idx + 1


def _extract_trajectories(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("trajectories"), list):
            return payload["trajectories"]
        if isinstance(payload.get("trajectory"), dict):
            return [payload["trajectory"]]
    return []


def _load_trajectory_file(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            payload = pickle.load(f)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        return []
    return _extract_trajectories(payload)


def _candidate_raw_trajectory_dirs() -> List[str]:
    workspace_root = Path(__file__).resolve().parents[3]
    return [
        str(workspace_root / "AutumnWeb" / "backend" / "trajectory_cache"),
        str(workspace_root / "AutumnWeb" / "backend" / "trajectories_cache"),
    ]


def _obfuscated_trajectory_dir() -> str:
    workspace_root = Path(__file__).resolve().parents[3]
    return str(
        workspace_root
        / "MARAProtocol"
        / "python_examples"
        / "autumnbench"
        / "example_benchmark"
        / "trajectories_obfuscated"
    )


def _looks_like_env_file(filename: str, env_name: str) -> bool:
    if filename.startswith(env_name):
        return True
    return f"{env_name}(" in filename or f"{env_name}_" in filename


def _load_prior_trajectories_for_env(
    env_name: str,
    use_obfuscation: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if use_obfuscation:
        directory = _obfuscated_trajectory_dir()
        source_type = "obfuscated"
    else:
        directory = ""
        source_type = "raw"
        for c in _candidate_raw_trajectory_dirs():
            if os.path.isdir(c):
                directory = c
                break

    meta: Dict[str, Any] = {
        "source_type": source_type,
        "directory": directory,
        "env_name": env_name,
    }
    if not directory or not os.path.isdir(directory):
        meta["status"] = "missing_directory"
        return [], meta

    selected_files: List[str] = []
    for name in sorted(os.listdir(directory)):
        if not (name.endswith(".pkl") or name.endswith(".json")):
            continue
        if _looks_like_env_file(name, env_name):
            selected_files.append(os.path.join(directory, name))

    trajectories: List[Dict[str, Any]] = []
    for fp in selected_files:
        try:
            trajectories.extend(_load_trajectory_file(fp))
        except Exception as e:
            logger.warning("Failed to load prior trajectory file %s: %s", fp, e)

    meta["status"] = "ok"
    meta["matched_files"] = selected_files
    meta["num_loaded_trajectories"] = len(trajectories)
    return trajectories, meta


def _evaluate_on_prior_trajectories(
    code: str,
    env_name: str,
    use_obfuscation: bool,
) -> Dict[str, Any]:
    trajectories, meta = _load_prior_trajectories_for_env(
        env_name=env_name,
        use_obfuscation=use_obfuscation,
    )
    if not trajectories:
        return {
            **meta,
            "evaluated": False,
            "reason": "no_trajectories",
        }
    result = execute_code_on_trajectories(code, trajectories)
    if not result.get("success"):
        return {
            **meta,
            "evaluated": True,
            "success": False,
            "error": result.get("error_info", {}).get("error_message", "Unknown error"),
        }
    results_list = result.get("results", [])
    total_correct = sum(r.get("correct", 0) for r in results_list)
    total_frames = sum(r.get("total", 0) for r in results_list)
    accuracy = total_correct / total_frames if total_frames > 0 else 1.0
    return {
        **meta,
        "evaluated": True,
        "success": True,
        "accuracy": accuracy,
        "total_correct": total_correct,
        "total_frames": total_frames,
        "num_trajectories": len(results_list),
    }

# ============================================================
# Prompt Constants
# ============================================================

PREDICT_DYNAMICS_SPEC = """\
Your task is to write or refine a Python function named `predict_dynamics`.

This function receives three arguments:
1. `visible_state` (dict): The observable state at the current timestep.
2. `hidden_state` (any Python object, or `None`): A state you define to track latent \
variables. For the *first* step in a trajectory this is `None`; your code must \
handle initialization.
3. `action` (dict): The action taken, e.g. `{"type": "click", "x": 3, "y": 5}` \
or `{"type": "left"}`.

It must return a tuple `(new_visible_state, new_hidden_state)`.

`visible_state` example (structure only; actual values vary):
```python
{
    'object_type_a': [{'position': {'x': 10, 'y': 5}, 'color': 'red'}],
    'object_type_b': [{'position': {'x': 3, 'y': 4}, 'color': 'blue'}, ...],
    'GRID_SIZE': 20
}
```

`action` examples:
```python
{'type': 'click', 'x': 7, 'y': 6}
{'type': 'noop'}
{'type': 'left'}
```

Output the complete `predict_dynamics` function inside a single ```python code block.
"""

INITIAL_GENERATION_PROMPT = """\
Here are observed trajectories showing the system's behavior.  Each trajectory \
shows State -> Action -> New State sequences.

{trajectories_text}
{qa_section}\
Based on these observations, write the `predict_dynamics` function from scratch.
Think carefully about:
- What objects exist and how they move/change
- Whether hidden state is needed (e.g., velocities, internal counters, modes)
- How different actions affect the state
- Edge cases such as boundary collisions
"""

REFINE_ERROR_PROMPT = """\
The current code was executed on all collected trajectories using **rollout mode** \
(each step uses the PREDICTED state from the previous step, not ground truth).

**Current Code:**
```python
{code}
```

**Overall Accuracy: {accuracy:.1%}** ({correct}/{total} frames correct)

**First error in each trajectory with errors:**
(Only the first mismatched frame per trajectory is shown.  Since rollout mode \
propagates errors, an early mistake cascades to all later frames.)

IMPORTANT — Regression detection: Compare the frame indices below with previous \
iterations.  If a trajectory's first-error frame index DECREASED compared to a \
previous iteration, a REGRESSION occurred — your last fix broke something that \
previously worked.  Prioritize fixing regressions.

{error_frames_text}

Please provide the complete updated `predict_dynamics` function that fixes these errors.
"""

REFINE_CRASH_PROMPT = """\
The current code failed to execute.

**Current Code:**
```python
{code}
```

**Error:**
```
{error_text}
```

Please fix the error and provide the complete updated `predict_dynamics` function.
"""

QUESTION_GENERATION_PROMPT = """\
Despite {n_iterations} refinement iterations, the code still has errors.

**Current Accuracy: {accuracy:.1%}**

**Persistent error frames:**
{error_frames_text}

Based on these persistent errors, what is the single most important experiment or \
observation that would help understand the dynamics better?

Propose exactly ONE specific, actionable question that an explorer agent should \
investigate in the environment.  The question should target the most critical \
aspect of the dynamics that the errors reveal is not yet understood.

Output the question on a single line, prefixed with "Q: ".
"""

TARGETED_EXPLORATION_PROMPT = """\
You previously wrote a `predict_dynamics` function, but it did not achieve \
100% accuracy and you raised a question about the dynamics.  A targeted \
exploration has been carried out to answer that question and collect \
additional data.

**Current code:**
```python
{current_code}
```

**Trajectories collected during targeted exploration:**
{trajectories_text}
{new_qa_section}\
Use these findings together with the error feedback that follows to \
refine the code.  The code will be re-evaluated on ALL trajectories \
(including the newly collected ones).
"""

QA_CONFIDENCE_SYSTEM_PROMPT = """\
You are evaluating the reliability of scientific Q&A knowledge used for dynamics modeling.

You will receive:
1) The current code and latest evaluation result.
2) Evaluation history from all refine loops in this main iteration.
3) A list of Q&A items with stable question_id fields.

For each question_id, output:
- score in [0, 1], where 1 means highly likely correct and 0 means highly likely incorrect.

Requirements:
- Be conservative and evidence-based.
- Use the full evaluation history, not only the latest result.
- Do not invent question_id values.
- Only output entries for question_id values where you have enough evidence to judge correctness.
"""

QA_CONFIDENCE_USER_PROMPT = """\
Current code:
```python
{code}
```

Current evaluation:
- accuracy: {accuracy:.4f} ({correct}/{total})
- first_error_frames:
{error_frames_text}

Evaluation history in this main loop:
{eval_history_text}

Q&A knowledge to score:
{qa_text}

Return one rating for each question_id above.
"""


_QA_CONFIDENCE_DECAY = 0.995
_QA_CONFIDENCE_WEIGHT = 2.0


# ============================================================
# Code Execution Utilities (adapted from AutumnLab code_executor.py)
# ============================================================

class _TimeoutException(Exception):
    pass


class _WriteOnlyStringIO(io.StringIO):
    """StringIO that raises on read — used to prevent user code from reading stdin."""
    def read(self, *args, **kwargs):
        raise OSError
    def readline(self, *args, **kwargs):
        raise OSError
    def readlines(self, *args, **kwargs):
        raise OSError
    def readable(self, *args, **kwargs):
        return False


class _RedirectStdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def _time_limit(seconds: float):
    def handler(signum, frame):
        raise _TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _RedirectStdin(stream):
                yield stream


def _eval_code(code: str, timeout: float = 3.0, exec_globals: dict = None,
               return_exec_globals: bool = False):
    """Execute *code* in a sandbox with I/O redirection and timeout."""
    try:
        exec_globals = {} if exec_globals is None else exec_globals
        with _swallow_io():
            with _time_limit(timeout):
                exec(code, exec_globals)
        return exec_globals if return_exec_globals else "passed"
    except _TimeoutException:
        return "timed out"
    except BaseException:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        extracted_tb = traceback.extract_tb(exc_traceback)
        exec_tb_frames = [f for f in extracted_tb if f.filename == "<string>"]
        if exec_tb_frames:
            exc_str = (
                "Traceback (most recent call last):\n"
                + "".join(traceback.format_list(exec_tb_frames))
                + "".join(traceback.format_exception_only(exc_type, exc_value))
            )
        else:
            exc_str = "".join(traceback.format_exception_only(exc_type, exc_value))
        return f"failed: {exc_str}"


def _convert_sets_to_lists(obj):
    """Recursively convert sets to sorted lists for JSON serialization."""
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if isinstance(obj, list):
        return [_convert_sets_to_lists(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _convert_sets_to_lists(v) for k, v in obj.items()}
    return obj


def _state_to_grid(state: Dict[str, Any]) -> Optional[List[List[str]]]:
    """Convert a rawFrame dict to a 2-D grid of color strings."""
    if not state or "GRID_SIZE" not in state:
        return None
    gs = state["GRID_SIZE"]
    grid = [["black"] * gs for _ in range(gs)]
    keys = sorted(state.keys())
    if "background" in keys:
        keys.insert(0, keys.pop(keys.index("background")))
    for k in keys:
        if k == "GRID_SIZE":
            continue
        objects = state.get(k)
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if isinstance(obj, dict) and "position" in obj and "color" in obj:
                pos = obj["position"]
                if isinstance(pos, dict) and "x" in pos and "y" in pos:
                    x, y, c = pos["x"], pos["y"], obj["color"]
                    if 0 <= y < gs and 0 <= x < gs:
                        grid[y][x] = c
    return grid


def _compare_visible_states(s1, s2) -> bool:
    """Compare two visible states by rendering them to grids."""
    if s1 is None and s2 is None:
        return True
    if s1 is None or s2 is None:
        return False
    g1, g2 = _state_to_grid(s1), _state_to_grid(s2)
    if g1 is None or g2 is None:
        return g1 == g2
    return g1 == g2


def _execute_worker(code_str: str, trajectories: List[Dict[str, Any]],
                    queue: multiprocessing.Queue):
    """Run predict_dynamics on trajectories in a subprocess (rollout mode)."""
    try:
        exec_globals = _eval_code(code_str, return_exec_globals=True)
        if isinstance(exec_globals, str):
            queue.put({"success": False, "error_info": {
                "error_message": exec_globals, "type": "compilation"}})
            return

        predict_fn = exec_globals.get("predict_dynamics")
        if not predict_fn or not callable(predict_fn):
            queue.put({"success": False, "error_info": {
                "error_message": "Function 'predict_dynamics' not found or not callable.",
                "type": "compilation"}})
            return

        all_results = []
        for traj in trajectories:
            hidden_state = None
            traj_results = []
            correct = 0
            total = 0
            frames = traj.get("frames", [])
            actions = traj.get("frameActions", [])
            last_predicted = None

            for i, action_info in enumerate(actions):
                if i + 1 >= len(frames):
                    break

                visible = (last_predicted if last_predicted is not None
                           else frames[i].get("rawFrame"))

                call_globals = {
                    "predict_dynamics_func": predict_fn,
                    "visible_state": copy.deepcopy(visible),
                    "hidden_state": copy.deepcopy(hidden_state),
                    "action": action_info,
                }
                code_to_exec = (
                    "__result__ = predict_dynamics_func("
                    "visible_state, hidden_state, action)"
                )
                result_or_error = _eval_code(
                    code_to_exec, exec_globals=call_globals,
                    return_exec_globals=True)

                if isinstance(result_or_error, str):
                    queue.put({"success": False, "error_info": {
                        "error_message": result_or_error,
                        "type": "runtime",
                        "inputs": {
                            "visible_state": visible,
                            "hidden_state": hidden_state,
                            "action": action_info,
                        }}})
                    return

                new_visible, hidden_state = result_or_error["__result__"]
                last_predicted = new_visible

                gt_next = frames[i + 1].get("rawFrame")
                is_correct = _compare_visible_states(new_visible, gt_next)
                if is_correct:
                    correct += 1
                total += 1

                traj_results.append({
                    "step_idx": i,
                    "input_visible_state": visible,
                    "predicted_visible_state": new_visible,
                    "ground_truth_next_state": gt_next,
                    "final_hidden_state": _convert_sets_to_lists(
                        copy.deepcopy(hidden_state)),
                    "action": action_info,
                    "is_correct": is_correct,
                })

            accuracy = correct / total if total > 0 else 1.0
            all_results.append({
                "steps": traj_results,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            })

        queue.put({"success": True, "results": all_results})
    except Exception as e:
        queue.put({"success": False, "error_info": {
            "error_message": f"Worker crash: {e}",
            "type": "worker_crash",
            "traceback": traceback.format_exc(),
        }})


def execute_code_on_trajectories(
    code_str: str, trajectories: List[Dict[str, Any]], timeout: float = 120.0,
) -> Dict[str, Any]:
    """Execute predict_dynamics code on trajectories in a sandboxed subprocess."""
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_execute_worker, args=(code_str, trajectories, queue))
    proc.start()
    try:
        result = queue.get(timeout=timeout)
    except QueueEmpty:
        result = {"success": False, "error_info": {
            "error_message": f"Execution timed out after {timeout}s.",
            "type": "timeout"}}
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join()
    return result


# ============================================================
# Helper Functions
# ============================================================

def extract_python_code(text: str) -> Optional[str]:
    """Extract the first ```python code block from *text*."""
    lines: List[str] = []
    in_block = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_block:
                return "\n".join(lines)
            in_block = True
        elif in_block:
            lines.append(line)
    if in_block and lines:
        return "\n".join(lines)
    return None


def format_trajectories_text(trajectories: List[Dict[str, Any]]) -> str:
    """Render trajectories as human-readable text for prompts."""
    parts: List[str] = []
    for i, traj in enumerate(trajectories):
        parts.append(f"--- Trajectory {i + 1} ---")
        frames = traj.get("frames", [])
        actions = traj.get("frameActions", [])
        for j, action in enumerate(actions):
            if j < len(frames):
                parts.append(f"State {j}: {json.dumps(frames[j].get('rawFrame'), sort_keys=True)}")
            a_type = action.get("type", "?") if isinstance(action, dict) else str(action)
            if a_type == "click":
                parts.append(f"Action {j}: click (x={action.get('x')}, y={action.get('y')})")
            else:
                parts.append(f"Action {j}: {a_type}")
            if j + 1 < len(frames):
                parts.append(
                    f"New State {j + 1}: "
                    f"{json.dumps(frames[j + 1].get('rawFrame'), sort_keys=True)}"
                )
        if len(frames) > len(actions) and len(actions) == 0:
            parts.append(
                f"Initial State: "
                f"{json.dumps(frames[0].get('rawFrame'), sort_keys=True)}"
            )
        parts.append("")
    return "\n".join(parts)


def find_first_error_frames(
    execution_results: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each trajectory, find the first incorrect step (if any).

    Returns a list of dicts with keys:
      traj_idx, frame_idx, total_frames, input_visible_state,
      hidden_state, action, predicted_next, ground_truth_next
    """
    errors: List[Dict[str, Any]] = []
    for t_idx, result in enumerate(execution_results):
        steps = result.get("steps", [])
        total = result.get("total", len(steps))
        for step in steps:
            if not step.get("is_correct"):
                errors.append({
                    "traj_idx": t_idx,
                    "frame_idx": step["step_idx"],
                    "total_frames": total,
                    "input_visible_state": step["input_visible_state"],
                    "hidden_state": step["final_hidden_state"],
                    "action": step["action"],
                    "predicted_next": step["predicted_visible_state"],
                    "ground_truth_next": step["ground_truth_next_state"],
                })
                break  # only first error per trajectory
    return errors


def format_error_frames_text(error_frames: List[Dict[str, Any]]) -> str:
    """Render first-error frames as text for the refine prompt."""
    if not error_frames:
        return "(no errors)"
    parts: List[str] = []
    for ef in error_frames:
        parts.append(
            f"Trajectory {ef['traj_idx'] + 1}, "
            f"Frame {ef['frame_idx']} / {ef['total_frames']}:"
        )
        parts.append(f"  Input visible state: {json.dumps(ef['input_visible_state'], sort_keys=True)}")
        parts.append(f"  Hidden state (predicted by code): {json.dumps(ef['hidden_state'], sort_keys=True)}")
        a = ef["action"]
        if isinstance(a, dict) and a.get("type") == "click":
            parts.append(f"  Action: click x={a.get('x')} y={a.get('y')}")
        else:
            parts.append(f"  Action: {a}")
        parts.append(f"  Code predicted next state: {json.dumps(ef['predicted_next'], sort_keys=True)}")
        parts.append(f"  Actual next state:         {json.dumps(ef['ground_truth_next'], sort_keys=True)}")
        parts.append("")
    return "\n".join(parts)


# ============================================================
# Q&A Formatting Helpers
# ============================================================

def format_qa_text(qa_list: List[Dict[str, Any]], label: str = "Knowledge") -> str:
    """Format Q&A pairs into a prompt section.  Returns empty string if no Q&A."""
    if not qa_list:
        return ""
    parts: List[str] = [f"\n**{label} from exploration (Q&A):**"]
    parts.append(
        "(Each QA includes a stable `question_id` used only for identity tracking "
        "across iterations.)"
    )
    parts.append(
        "(`confidence` is in [0,1]. Higher means the QA is more reliable; "
        "0 means the QA is very likely incorrect and should generally not be relied on.)"
    )
    for i, qa in enumerate(qa_list, 1):
        conf = qa.get("confidence")
        conf_text = f" [confidence={float(conf):.3f}]" if isinstance(conf, (int, float)) else ""
        question_id = qa.get("question_id", qa.get("qa_id", f"question_{i}"))
        parts.append(f"Q{i} [question_id={question_id}]: {qa.get('question', '?')}")
        parts.append(f"A{i}{conf_text}: {qa.get('answer', '?')}")
        parts.append("")
    return "\n".join(parts) + "\n"


def _normalize_qa_entry(qa: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    question_id = qa.get("question_id", qa.get("qa_id"))
    if not question_id:
        digest = hashlib.sha1(f"{question}|{answer}|{idx}".encode("utf-8")).hexdigest()[:12]
        question_id = f"question_{digest}"

    alpha = float(qa.get("alpha", 1.0))
    beta = float(qa.get("beta", 1.0))
    conf = qa.get("confidence")
    if not isinstance(conf, (int, float)):
        conf = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

    history = qa.get("confidence_history")
    if not isinstance(history, list):
        history = []

    return {
        **qa,
        "question_id": str(question_id),
        "question": question,
        "answer": answer,
        "alpha": alpha,
        "beta": beta,
        "confidence": float(conf),
        "confidence_history": history,
    }


def _normalize_qa_list(qa_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_normalize_qa_entry(qa, idx) for idx, qa in enumerate(qa_list)]


def _build_eval_history_entry(state: Dict[str, Any], status: str) -> Dict[str, Any]:
    code = state.get("code") or ""
    return {
        "refine_iteration": int(state.get("refine_iteration", 0)),
        "refine_eval_index": int(state.get("refine_eval_index", 0)),
        "status": status,
        "code": code,
    }


def _format_eval_history_text(eval_history: List[Dict[str, Any]]) -> str:
    if not eval_history:
        return "(none)"
    parts: List[str] = []
    for row in eval_history:
        parts.append(
            "=== Eval iter={iter_}, eval={eval_}, status={status} ===".format(
                iter_=row.get("refine_iteration", "?"),
                eval_=row.get("refine_eval_index", "?"),
                status=row.get("status", "?"),
            )
        )
        parts.append("Code:")
        parts.append("```python")
        parts.append(str(row.get("code", "")))
        parts.append("```")
        if row.get("status") == "success":
            parts.append("Execution result:")
            parts.append(
                "- accuracy={acc:.4f} ({c}/{t})".format(
                    acc=float(row.get("accuracy", 0.0)),
                    c=int(row.get("total_correct", 0)),
                    t=int(row.get("total_frames", 0)),
                )
            )
            parts.append("- first_error_frames:")
            parts.append(str(row.get("first_error_frames_text", "(none)")))
        else:
            parts.append("Execution result:")
            parts.append("- execution_error:")
            parts.append(str(row.get("error", "unknown")))
        parts.append("")
    return "\n".join(parts)


class QAConfidenceRating(BaseModel):
    question_id: str = Field(description="Stable QA identifier")
    score: float = Field(ge=0.0, le=1.0, description="Correctness confidence score in [0,1]")
    rationale: str = Field(default="", description="Short reason")


class QAConfidenceRatings(BaseModel):
    ratings: List[QAConfidenceRating] = Field(
        default_factory=list,
        description="Confidence ratings for QA entries",
    )


# ============================================================
# State Definition
# ============================================================

class RefineState(TypedDict):
    code: str                                # Current predict_dynamics source
    all_trajectories: list                   # All trajectories to evaluate against
    targeted_trajectories: list              # Trajectories from targeted exploration
    from_targeted_exploration: bool          # Whether to present targeted-exploration results first
    conversation_history: list               # List[BaseMessage] — persisted across invocations
    refine_iteration: int                    # How many refine LLM calls so far this round
    max_refine_iterations: int               # Limit on refine iterations
    accuracy: float                          # Latest overall accuracy
    total_correct: int
    total_frames: int
    execution_error: Optional[str]           # Error message if code crashed
    per_trajectory_results: list             # Raw execution results per trajectory
    first_error_frames: list                 # First error frame per trajectory
    questions: list                          # Questions for the explorer (if not 100%)
    code_perfect: bool                       # True if accuracy == 1.0
    is_first_generation: bool                # True if no code exists yet
    new_qa: list                             # New Q&A pairs from latest exploration (highlighted)
    all_qa: list                             # All historical Q&A pairs (background knowledge)
    code_refine_log_dir: str                 # Log directory for this main-loop code_refine
    main_loop_index: int                     # Main loop index from baseline graph
    refine_eval_index: int                   # Index for each execute_and_evaluate pass
    env_name: str                            # Environment id (e.g., 7XF97)
    use_obfuscation: bool                    # Whether current run uses obfuscated observations
    prior_trajectory_eval: dict              # Metrics on historical trajectories
    eval_history_this_round: list            # All execute/evaluate outcomes in this main-loop round


# ============================================================
# Graph Builder
# ============================================================

def create_refine_graph(llm):
    """Build and compile the refinement subgraph.

    Args:
        llm: A LangChain chat model instance (e.g. ChatOpenAI).

    Returns:
        A compiled LangGraph that accepts and returns RefineState.
    """

    # ----------------------------------------------------------
    # Node: initial code generation from trajectories
    # ----------------------------------------------------------
    def initial_generate_node(state: RefineState) -> dict:
        logger.info("RefineGraph: initial code generation")
        traj_text = format_trajectories_text(state["all_trajectories"])
        qa_section = format_qa_text(
            state.get("all_qa") or [], label="Background knowledge")
        user_msg = HumanMessage(content=INITIAL_GENERATION_PROMPT.format(
            trajectories_text=traj_text, qa_section=qa_section))

        history = list(state.get("conversation_history") or [])
        if not history:
            history.append(SystemMessage(content=PREDICT_DYNAMICS_SPEC))
        history.append(user_msg)

        response = llm.invoke(history)
        ai_msg = AIMessage(content=response.content)
        history.append(ai_msg)
        _log_refine_llm_call(
            state=state,
            node_name="initial_generate",
            llm_input={"history": history[:-1]},
            llm_output={"response": response.content},
        )

        extracted = extract_python_code(response.content)
        return {
            "code": extracted or "",
            "conversation_history": history,
            "is_first_generation": False,
        }

    # ----------------------------------------------------------
    # Node: present findings from targeted exploration
    # ----------------------------------------------------------
    def present_targeted_findings_node(state: RefineState) -> dict:
        logger.info("RefineGraph: presenting targeted-exploration findings")
        tgt_trajs = state.get("targeted_trajectories") or []
        traj_text = format_trajectories_text(tgt_trajs) if tgt_trajs else "(none)"
        new_qa_section = format_qa_text(
            state.get("new_qa") or [], label="Answers to your earlier question")
        msg = HumanMessage(content=TARGETED_EXPLORATION_PROMPT.format(
            current_code=state.get("code") or "(none)",
            trajectories_text=traj_text,
            new_qa_section=new_qa_section,
        ))
        history = list(state.get("conversation_history") or [])
        if not history:
            history.append(SystemMessage(content=PREDICT_DYNAMICS_SPEC))
        history.append(msg)
        return {
            "conversation_history": history,
            "from_targeted_exploration": False,
        }

    # ----------------------------------------------------------
    # Node: execute code on all trajectories, compute accuracy
    # ----------------------------------------------------------
    def execute_and_evaluate_node(state: RefineState) -> dict:
        code = state["code"]
        trajs = state["all_trajectories"]
        logger.info("RefineGraph: executing code on %d trajectories", len(trajs))
        prior_eval: Dict[str, Any] = {}
        eval_history = list(state.get("eval_history_this_round") or [])

        if not code or not code.strip():
            next_eval_idx = _persist_refine_eval_error(
                state, "No code to execute.", prior_eval=prior_eval)
            history_entry = _build_eval_history_entry(state, status="execution_error")
            history_entry["error"] = "No code to execute."
            eval_history.append(history_entry)
            return {
                "accuracy": 0.0,
                "total_correct": 0,
                "total_frames": 0,
                "execution_error": "No code to execute.",
                "per_trajectory_results": [],
                "first_error_frames": [],
                "code_perfect": False,
                "refine_eval_index": next_eval_idx,
                "eval_history_this_round": eval_history,
            }

        result = execute_code_on_trajectories(code, trajs)
        prior_eval = _evaluate_on_prior_trajectories(
            code=code,
            env_name=state.get("env_name", ""),
            use_obfuscation=bool(state.get("use_obfuscation", False)),
        )
        if prior_eval.get("evaluated") and prior_eval.get("success"):
            logger.info(
                "RefineGraph: prior trajectories accuracy = %.1f%% (%d/%d) from %s",
                prior_eval["accuracy"] * 100,
                prior_eval["total_correct"],
                prior_eval["total_frames"],
                prior_eval.get("directory"),
            )
        else:
            logger.info(
                "RefineGraph: prior trajectories eval skipped/failed (%s)",
                prior_eval.get("reason") or prior_eval.get("error") or prior_eval.get("status"),
            )

        if not result["success"]:
            error_info = result.get("error_info", {})
            error_msg = error_info.get("error_message", "Unknown error")
            logger.warning("RefineGraph: execution failed — %s", error_msg)
            next_eval_idx = _persist_refine_eval_error(
                state, error_msg, prior_eval=prior_eval)
            history_entry = _build_eval_history_entry(state, status="execution_error")
            history_entry["error"] = error_msg
            eval_history.append(history_entry)
            return {
                "accuracy": 0.0,
                "total_correct": 0,
                "total_frames": 0,
                "execution_error": error_msg,
                "per_trajectory_results": [],
                "first_error_frames": [],
                "code_perfect": False,
                "refine_eval_index": next_eval_idx,
                "eval_history_this_round": eval_history,
            }

        results_list = result["results"]
        total_correct = sum(r["correct"] for r in results_list)
        total_frames = sum(r["total"] for r in results_list)
        accuracy = total_correct / total_frames if total_frames > 0 else 1.0

        first_errors = find_first_error_frames(results_list, trajs)

        logger.info(
            "RefineGraph: accuracy = %.1f%% (%d/%d)",
            accuracy * 100, total_correct, total_frames,
        )
        next_eval_idx = _persist_refine_eval_artifacts(
            state=state,
            results_list=results_list,
            accuracy=accuracy,
            total_correct=total_correct,
            total_frames=total_frames,
            prior_eval=prior_eval,
        )
        history_entry = _build_eval_history_entry(state, status="success")
        history_entry.update({
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_frames": total_frames,
            "first_error_frames_text": format_error_frames_text(first_errors),
        })
        eval_history.append(history_entry)

        return {
            "accuracy": accuracy,
            "total_correct": total_correct,
            "total_frames": total_frames,
            "execution_error": None,
            "per_trajectory_results": results_list,
            "first_error_frames": first_errors,
            "code_perfect": accuracy >= 1.0,
            "refine_eval_index": next_eval_idx,
            "prior_trajectory_eval": prior_eval,
            "eval_history_this_round": eval_history,
        }

    # ----------------------------------------------------------
    # Node: score QA reliability and update confidence state
    # ----------------------------------------------------------
    def score_qa_confidence_node(state: RefineState) -> dict:
        qa_list = _normalize_qa_list(list(state.get("all_qa") or []))
        if not qa_list:
            return {"all_qa": qa_list}

        if state.get("execution_error"):
            logger.info("RefineGraph: execution_error present, skip QA scoring")
            return {"all_qa": qa_list}

        code = state.get("code") or ""
        error_text = format_error_frames_text(state.get("first_error_frames") or [])
        qa_text = format_qa_text(qa_list, label="Q&A memory")
        eval_history_text = _format_eval_history_text(
            list(state.get("eval_history_this_round") or [])
        )
        llm_messages: List[Any] = [
            {"role": "system", "content": QA_CONFIDENCE_SYSTEM_PROMPT},
            {"role": "user", "content": QA_CONFIDENCE_USER_PROMPT.format(
                code=code,
                accuracy=float(state.get("accuracy", 0.0)),
                correct=int(state.get("total_correct", 0)),
                total=int(state.get("total_frames", 0)),
                error_frames_text=error_text,
                eval_history_text=eval_history_text,
                qa_text=qa_text,
            )},
        ]

        ratings_map: Dict[str, QAConfidenceRating] = {}
        try:
            structured_llm = llm.with_structured_output(QAConfidenceRatings)
            scored: QAConfidenceRatings = structured_llm.invoke(llm_messages)
            for rating in scored.ratings:
                ratings_map[rating.question_id] = rating
            _log_refine_llm_call(
                state=state,
                node_name="score_qa_confidence",
                llm_input={"messages": llm_messages},
                llm_output={"ratings": [
                    r.model_dump() if hasattr(r, "model_dump") else r.dict()
                    for r in scored.ratings
                ]},
            )
        except Exception as e:
            logger.warning("RefineGraph: QA confidence scoring failed: %s", e)
            _log_refine_llm_call(
                state=state,
                node_name="score_qa_confidence",
                llm_input={"messages": llm_messages},
                llm_output={},
                error=str(e),
            )
            return {"all_qa": qa_list}

        updated_qas: List[Dict[str, Any]] = []
        for qa in qa_list:
            rating = ratings_map.get(qa["question_id"])
            if not rating:
                updated_qas.append(qa)
                continue

            score = max(0.0, min(1.0, float(rating.score)))
            alpha = _QA_CONFIDENCE_DECAY * float(qa.get("alpha", 1.0)) + _QA_CONFIDENCE_WEIGHT * score
            beta = _QA_CONFIDENCE_DECAY * float(qa.get("beta", 1.0)) + _QA_CONFIDENCE_WEIGHT * (1.0 - score)
            confidence = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5

            history = list(qa.get("confidence_history") or [])
            history.append(score)
            if len(history) > 100:
                history = history[-100:]

            updated_qas.append({
                **qa,
                "alpha": alpha,
                "beta": beta,
                "confidence": confidence,
                "confidence_history": history,
                "last_confidence_score": score,
            })

        return {"all_qa": updated_qas}

    # ----------------------------------------------------------
    # Node: ask LLM to refine code based on first-error frames
    # ----------------------------------------------------------
    def refine_code_node(state: RefineState) -> dict:
        iteration = state["refine_iteration"] + 1
        logger.info("RefineGraph: refine iteration %d", iteration)

        code = state["code"]
        history = list(state.get("conversation_history") or [])

        # Build the human message depending on whether execution crashed
        if state.get("execution_error"):
            user_content = REFINE_CRASH_PROMPT.format(
                code=code,
                error_text=state["execution_error"],
            )
        else:
            error_text = format_error_frames_text(state["first_error_frames"])
            user_content = REFINE_ERROR_PROMPT.format(
                code=code,
                accuracy=state["accuracy"],
                correct=state["total_correct"],
                total=state["total_frames"],
                error_frames_text=error_text,
            )

        user_msg = HumanMessage(content=user_content)
        history.append(user_msg)

        response = llm.invoke(history)
        ai_msg = AIMessage(content=response.content)
        history.append(ai_msg)
        _log_refine_llm_call(
            state=state,
            node_name="refine_code",
            llm_input={"history": history[:-1]},
            llm_output={"response": response.content},
        )

        extracted = extract_python_code(response.content)

        return {
            "code": extracted or code,  # keep old code if extraction fails
            "conversation_history": history,
            "refine_iteration": iteration,
        }

    # ----------------------------------------------------------
    # Node: generate diagnostic questions for the explorer
    # ----------------------------------------------------------
    def generate_questions_node(state: RefineState) -> dict:
        logger.info("RefineGraph: generating diagnostic questions")
        error_text = format_error_frames_text(state["first_error_frames"])
        prompt = QUESTION_GENERATION_PROMPT.format(
            n_iterations=state["refine_iteration"],
            accuracy=state["accuracy"],
            error_frames_text=error_text,
        )
        history = list(state.get("conversation_history") or [])
        history.append(HumanMessage(content=prompt))

        response = llm.invoke(history)
        history.append(AIMessage(content=response.content))
        _log_refine_llm_call(
            state=state,
            node_name="generate_questions",
            llm_input={"history": history[:-1]},
            llm_output={"response": response.content},
        )

        # Parse Q: lines
        questions = []
        for line in response.content.split("\n"):
            line = line.strip()
            if line.startswith("Q:") or line.startswith("Q："):
                questions.append(line[2:].strip())
        if not questions:
            # Fallback: treat each non-empty line as a question
            questions = [l.strip() for l in response.content.split("\n")
                         if l.strip() and not l.strip().startswith("```")]

        return {
            "questions": questions,
            "conversation_history": history,
        }

    # ----------------------------------------------------------
    # Routing functions
    # ----------------------------------------------------------
    def route_entry(state: RefineState) -> str:
        if state.get("is_first_generation"):
            return "initial_generate"
        if state.get("from_targeted_exploration"):
            return "present_targeted_findings"
        return "execute_and_evaluate"

    def route_after_evaluate(state: RefineState) -> str:
        if state.get("code_perfect"):
            return END
        can_refine = state["refine_iteration"] < state["max_refine_iterations"]
        if can_refine:
            return "refine_code"
        return "generate_questions"

    # ----------------------------------------------------------
    # Build the graph
    # ----------------------------------------------------------
    workflow = StateGraph(RefineState)

    workflow.add_node("entry_router", lambda state: {})  # passthrough
    workflow.add_node("initial_generate", initial_generate_node)
    workflow.add_node("present_targeted_findings", present_targeted_findings_node)
    workflow.add_node("execute_and_evaluate", execute_and_evaluate_node)
    workflow.add_node("score_qa_confidence", score_qa_confidence_node)
    workflow.add_node("refine_code", refine_code_node)
    workflow.add_node("generate_questions", generate_questions_node)

    workflow.set_entry_point("entry_router")

    workflow.add_conditional_edges("entry_router", route_entry, {
        "initial_generate": "initial_generate",
        "present_targeted_findings": "present_targeted_findings",
        "execute_and_evaluate": "execute_and_evaluate",
    })

    workflow.add_edge("initial_generate", "execute_and_evaluate")
    workflow.add_edge("present_targeted_findings", "execute_and_evaluate")
    workflow.add_edge("execute_and_evaluate", "score_qa_confidence")

    workflow.add_conditional_edges("score_qa_confidence", route_after_evaluate, {
        "refine_code": "refine_code",
        "generate_questions": "generate_questions",
        END: END,
    })

    workflow.add_edge("refine_code", "execute_and_evaluate")
    workflow.add_edge("generate_questions", END)

    return workflow.compile()
