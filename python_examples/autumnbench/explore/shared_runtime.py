"""
Shared state and runtime helpers for the new explore workflow.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

# Reuse existing environment wrapper and trajectory conversion implementation.
_FILE_DIR = Path(__file__).resolve().parent
_AUTUMNBENCH_DIR = _FILE_DIR.parent
_REPO_ROOT = _AUTUMNBENCH_DIR.parents[2]  # /app
for _p in [str(_AUTUMNBENCH_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_graph import ExplorationEnvironment, convert_env_trajectory  # noqa: E402

logger = logging.getLogger(__name__)


class TrajectoryRecord(TypedDict, total=False):
    frames: list
    frameActions: list
    traj_path: str
    code_path: str
    code_description: str
    trajectory_description: str
    code_content: str
    source_workspace_dir: str


class ProblemSample(TypedDict):
    trajectory: TrajectoryRecord
    error: Dict[str, Any]


class ProblemItem(TypedDict):
    problem_id: str
    summary: str
    problem_samples: List[ProblemSample]
    tried: bool
    solved: bool


class GlobalConfig(TypedDict):
    env_name: str
    data_dir: str
    llm_model: str
    use_obfuscation: bool


class BudgetConfig(TypedDict):
    max_explore_steps: int


class PerfectZoneConfig(TypedDict):
    collect_agent_max_turns: int
    shell_command_timeout_seconds: int
    max_perfect_zone_rounds: int


class ProblemRepairConfig(TypedDict):
    collect_agent_max_turns: int
    shell_command_timeout_seconds: int
    shell_refine_max_turns: int
    cleanup_targeted_runtime: bool


class TargetedCollectionConfig(TypedDict):
    collect_agent_max_turns: int
    shell_command_timeout_seconds: int
    cleanup_targeted_runtime: bool


class WorkflowConfig(TypedDict):
    global_config: GlobalConfig
    budget: BudgetConfig
    perfect_zone: PerfectZoneConfig
    problem_repair: ProblemRepairConfig
    targeted_collection: TargetedCollectionConfig


class WorkflowState(TypedDict):
    # Core modeling state.
    code: str
    config: WorkflowConfig

    # Trajectory and error memory.
    all_trajectories: List[TrajectoryRecord]
    correct_trajectories: List[TrajectoryRecord]
    # Newly collected trajectories from the latest perfect-zone pass.
    new_trajectories: List[TrajectoryRecord]
    error_samples: list
    # None means current code has not been analyzed yet (or analysis is stale).
    # [] means analyzed and no problems remain.
    problems: Optional[List[ProblemItem]]
    target_problem: Optional[ProblemItem]

    # Control and budget.
    target_problem_fixed: bool
    should_terminate_workflow: bool
    need_more_targeted_trajectories: bool
    # Whether current code still leaves unexplained frames after validation.
    has_unexplained_frames: bool
    coding_can_explain: bool

    # Experiment logs.
    experiment_run_dir: str
    current_loop_dir: str
    explore_log_dir: str
    refine_log_dir: str

    # Transient repair-session fields (not global truth).
    repair_runtime_key: Optional[str]
    session_code_candidate: Optional[str]

    # Logging directory passed down the graph
    current_log_dir: str


def find_program_path(data_dir: str, env_name: str) -> str:
    for subdir in ["tests", "programs", ""]:
        candidate = os.path.join(data_dir, subdir, f"{env_name}.sexp")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Cannot find {env_name}.sexp under {data_dir}")


def parse_action_string(action_text: str) -> Optional[Dict[str, Any]]:
    first_line = (action_text or "").strip().split("\n")[0].strip().lower()
    if first_line in ("left", "right", "up", "down", "noop"):
        return {"type": first_line}
    m = re.match(r"click\s+(\d+)\s+(\d+)", first_line)
    if m:
        return {"type": "click", "x": int(m.group(1)), "y": int(m.group(2))}
    return None


def run_action_sequence_collect_trajectory(
    env: ExplorationEnvironment,
    actions: List[str],
) -> Dict[str, Any]:
    env.reset()
    for act in actions:
        env.step(act)
    raw_traj = env.get_raw_trajectory()
    return convert_env_trajectory(raw_traj)


def build_problem_item(
    problem_id: str,
    summary: str,
    problem_samples: List[ProblemSample],
) -> ProblemItem:
    return {
        "problem_id": problem_id,
        "summary": summary,
        "problem_samples": list(problem_samples),
        "tried": False,
        "solved": False,
    }


def append_problem_samples(
    problem: ProblemItem,
    new_samples: List[ProblemSample],
) -> ProblemItem:
    updated = dict(problem) # type: ignore
    pool = list(updated.get("problem_samples", []))
    for sample in list(new_samples or []):
        traj = sample.get("trajectory") if isinstance(sample, dict) else None
        err = sample.get("error") if isinstance(sample, dict) else None
        if isinstance(traj, dict) and isinstance(err, dict):
            # 浅拷贝样本字典，避免修改原始样本
            pool.append({"trajectory": traj, "error": err})
    updated["problem_samples"] = pool # type: ignore
    return updated # type: ignore


def build_problem_samples_from_errors(
    trajectories: List[TrajectoryRecord],
    matched_problem_errors: List[Dict[str, Any]],
) -> List[ProblemSample]:
    errors_by_idx: Dict[int, Dict[str, Any]] = {}
    for err in list(matched_problem_errors or []):
        if not isinstance(err, dict):
            continue
        idx = err.get("traj_idx")
        if isinstance(idx, int) and 0 <= idx < len(trajectories):
            # 浅拷贝 error 字典，并去除 traj_idx
            errors_by_idx[idx] = {k: v for k, v in err.items() if k != "traj_idx"}
    out: List[ProblemSample] = []
    for idx, traj in enumerate(trajectories):
        err = errors_by_idx.get(idx)
        if isinstance(traj, dict) and isinstance(err, dict):
            # 组装新的 ProblemSample 字典
            out.append({"trajectory": traj, "error": err})
    return out


def get_problem_by_id(problems: List[ProblemItem], problem_id: str) -> Optional[ProblemItem]:
    for p in problems:
        if p.get("problem_id") == problem_id:
            return p
    return None


def extract_errors_by_traj_path(problem_samples: List[Any]) -> Dict[str, Dict[str, Any]]:
    errors_by_traj_path = {}
    for sample in problem_samples:
        if isinstance(sample, dict):
            traj = sample.get("trajectory")
            err = sample.get("error")
            if isinstance(traj, dict) and isinstance(err, dict):
                t_path = traj.get("traj_path")
                if t_path:
                    errors_by_traj_path[t_path] = err
    return errors_by_traj_path


def sync_trajectories_to_workspace(
    trajectories: List[TrajectoryRecord],
    prefix: str,
    target_workspace_dir: str,
    errors_by_traj_path: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    import os
    import shutil
    import json
    import glob
    
    # Find max index to avoid conflicts
    existing_files = glob.glob(os.path.join(target_workspace_dir, "traj", f"{prefix}_*.json"))
    max_idx = -1
    for f in existing_files:
        try:
            basename = os.path.basename(f)
            idx_str = basename.replace(f"{prefix}_", "").replace(".json", "")
            max_idx = max(max_idx, int(idx_str))
        except ValueError:
            pass
    start_idx = max_idx + 1
    
    trajectory_index = []
    for i, traj in enumerate(trajectories):
        idx = start_idx + i
        source_workspace = traj.get("source_workspace_dir")
        traj_path = traj.get("traj_path")
        code_path = traj.get("code_path")
        
        copied_traj = False
        copied_code = False
        
        target_traj_path = f"traj/{prefix}_{idx:03d}.json"
        if source_workspace and traj_path:
            src_traj_full = os.path.join(source_workspace, traj_path)
            dst_traj_full = os.path.join(target_workspace_dir, target_traj_path)
            if os.path.exists(src_traj_full):
                os.makedirs(os.path.dirname(dst_traj_full), exist_ok=True)
                shutil.copy2(src_traj_full, dst_traj_full)
                copied_traj = True
        
        if not copied_traj:
            from runtime_utils import trajectory_to_saved_payload
            payload = trajectory_to_saved_payload(traj)
            if payload:
                dst_traj_full = os.path.join(target_workspace_dir, target_traj_path)
                os.makedirs(os.path.dirname(dst_traj_full), exist_ok=True)
                with open(dst_traj_full, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=True, indent=2)
                copied_traj = True
        
        target_code_path = f"explore_code/{prefix}_{idx:03d}.py"
        if source_workspace and code_path:
            src_code_full = os.path.join(source_workspace, code_path)
            dst_code_full = os.path.join(target_workspace_dir, target_code_path)
            if os.path.exists(src_code_full):
                os.makedirs(os.path.dirname(dst_code_full), exist_ok=True)
                shutil.copy2(src_code_full, dst_code_full)
                copied_code = True
        
        if not copied_code and traj.get("code_content"):
            dst_code_full = os.path.join(target_workspace_dir, target_code_path)
            os.makedirs(os.path.dirname(dst_code_full), exist_ok=True)
            with open(dst_code_full, "w", encoding="utf-8") as f:
                f.write(traj["code_content"])
            copied_code = True
        
        if copied_traj:
            entry = {
                "trajectory_path": target_traj_path,
                "code_path": target_code_path if copied_code else None,
                "trajectory_description": traj.get("trajectory_description", ""),
                "code_description": traj.get("code_description", ""),
                "type": prefix
            }
            if errors_by_traj_path and traj_path and traj_path in errors_by_traj_path:
                entry["error"] = errors_by_traj_path[traj_path]
            trajectory_index.append(entry)
            
    return trajectory_index


def attach_metadata_to_trajectories(
    execute_run_command_fn: Callable[..., str],
    env_name: str,
    runtime_key: str,
    log_file_path: str,
    normalized_trajectories: List[TrajectoryRecord],
    workspace_dir: str,
) -> None:
    from runtime_utils import extract_stdout_text, read_code_from_runtime
    import json
    
    log_cmd = f"cat {log_file_path}"
    log_raw = execute_run_command_fn(
        log_cmd,
        env_name=env_name,
        timeout_seconds=10,
        runtime_key=runtime_key,
    )
    log_text = extract_stdout_text(log_raw).strip()
    
    metadata_by_traj = {}
    if log_text and not log_text.startswith("cat:"):
        for line in log_text.splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                t_path = entry.get("trajectory_path", "").strip()
                if t_path:
                    if t_path.startswith("/workspace/"):
                        t_path = t_path[len("/workspace/"):]
                    metadata_by_traj[t_path] = entry
            except Exception:
                pass

    code_contents = {}
    for entry in metadata_by_traj.values():
        c_path = entry.get("code_path", "").strip()
        if c_path and c_path not in code_contents:
            code_contents[c_path] = read_code_from_runtime(
                execute_run_command_fn=execute_run_command_fn,
                env_name=env_name,
                runtime_key=runtime_key,
                code_path=c_path,
                timeout_seconds=10,
            )

    for traj in normalized_trajectories:
        t_path = traj.get("traj_path", "")
        if t_path in metadata_by_traj:
            entry = metadata_by_traj[t_path]
            traj["code_path"] = entry.get("code_path", "")
            traj["code_description"] = entry.get("code_description", "")
            traj["trajectory_description"] = entry.get("trajectory_description", "")
            traj["code_content"] = code_contents.get(traj["code_path"], "")
        else:
            pass
            # Fallback for trajectories that the agent forgot to log
            # if t_path.startswith("traj/") and t_path.endswith(".json"):
            #     base_name = t_path[len("traj/"):-len(".json")]
            #     guessed_code_path = f"explore_code/{base_name}.py"
            #     traj["code_path"] = guessed_code_path
            #     traj["code_description"] = ""
            #     traj["trajectory_description"] = ""
            #     traj["code_content"] = read_code_from_runtime(
            #         execute_run_command_fn=execute_run_command_fn,
            #         env_name=env_name,
            #         runtime_key=runtime_key,
            #         code_path=guessed_code_path,
            #         timeout_seconds=10,
            #     )
        traj["source_workspace_dir"] = workspace_dir

