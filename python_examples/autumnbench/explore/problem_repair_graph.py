"""
Problem repair loop graph.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
import os
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph


from error_analysis_graph import run_error_analysis
from shared_runtime import (
    ProblemItem,
    ProblemSample,
    WorkflowConfig,
    WorkflowState,
    TrajectoryRecord,
    append_problem_samples,
    build_problem_samples_from_errors,
    sync_trajectories_to_workspace
)
from targeted_collection_graph import call_targeted_collection_graph_from_docker_runtime

from log_manager import create_subgraph_dir, create_node_dir, get_next_sequence_dir, NodeLoggingCallbackHandler

logger = logging.getLogger(__name__)

_FILE_DIR = Path(__file__).resolve().parent
_AUTUMNBENCH_DIR = _FILE_DIR.parent
_REPO_ROOT = _AUTUMNBENCH_DIR.parents[2]
for _p in [str(_AUTUMNBENCH_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_by_code.env_wrapper import execute_run_command  # noqa: E402
from explore_by_code.env_wrapper import get_langchain_tools  # noqa: E402
from explore_by_code.env_wrapper import get_or_create_runtime_info  # noqa: E402
from explore_by_code.env_wrapper import close_runtime  # noqa: E402
from runtime_utils import (  # noqa: E402
    load_all_trajectories_from_runtime,
    message_content_to_text,
    read_code_from_runtime,
    write_code_to_runtime,
    write_trajectories_to_runtime,
)


class ProblemRepairState(TypedDict):
    # Shared config
    config: WorkflowConfig

    # Active repair context for current target problem.
    code: str
    target_problem: ProblemItem
    correct_trajectories: List[TrajectoryRecord]
    all_trajectories: List[TrajectoryRecord]

    # Control and output flags produced within this subgraph.
    need_more_targeted_trajectories: bool
    session_code_candidate: str
    target_problem_fixed: bool

    # Runtime routing keys.
    repair_runtime_key: str
    targeted_runtime_key: str

    current_log_dir: str # Logging directory passed down the graph


_REFINE_SYSTEM_PROMPT = """\
You are an autonomous code-repair engineer working in /workspace with two tools:
- run_command_in_docker(command: str)
- collect_targeted_trajectories(reason: str = "")

Goal:
- Fix the current Python dynamics model so it reaches 100% accuracy on targeted trajectories without breaking previously correct trajectories.
- You may run any shell/Python experiments needed before editing code.

Environment basics:
- Deterministic GRID_SIZE x GRID_SIZE world.
- You can treat this as an MDP/POMDP-style dynamics problem: visible observations may
  be sufficient in some environments, while others require hidden_state to represent
  latent dynamics.
- Valid actions for env.step(action):
  - click x y: Click on the cell at location (x, y). For GRID_SIZE, x and y
    must each be between 0 and GRID_SIZE-1 inclusive. (Do NOT use commas, e.g., "click 3, 4" is invalid).
  - left: Press the left arrow key.
  - right: Press the right arrow key.
  - up: Press the up arrow key.
  - down: Press the down arrow key.
  - noop: Do nothing and continue to the next step.

Python API example (interface demonstration):
```python
from env_api_client import RemoteEnvWrapper
import json

env = RemoteEnvWrapper()
r = env.reset()
s = env.step('click 3 4')
t = env.save_trajectory('traj_name') # Saves to /workspace/traj
t2 = env.save_trajectory('tmp_traj_name', dir='tmp_traj') # Saves to /workspace/tmp_traj
```

RemoteEnvWrapper method semantics:
- reset() -> dict
  Initializes environment session and returns the initial visible state dict.
- step(action: str) -> dict
  Executes one valid action and returns the next visible state dict.
  For click actions use the exact format: "click x y" (for example: "click 3 4").
- save_trajectory(filename: Optional[str] = None, dir: Optional[str] = None) -> dict
  Persists the currently collected trajectory. By default, saves to /workspace/traj/...
  Use dir="tmp_traj" to save temporary experimental trajectories.
  Returns a success indicator object (for example: {"success": true}).

Workspace Structure & Background:
You are operating in /workspace. You can use shell commands to view the contents of the directory as needed.
Important directories and files:
- /workspace/candidate_model.py: The current Python dynamics model code that you need to fix. (If this file does not exist, it means this is the initial generation and you need to write it from scratch).
- /workspace/explore_code/: Directory containing Python scripts used to generate the formal trajectories.
- /workspace/traj/: Directory containing the formal trajectories. Files prefixed with `seed_pool_` are target problem trajectories to fix, while `correct_pool_` indicates trajectories that the current `candidate_model` can correctly predict, which you should try your best not to break.
- /workspace/trajectory_index.jsonl: A JSONL (JSON Lines) file containing objects describing available trajectories. Each line has the following format:
  {"trajectory_path": "traj/seed_pool_000.json", "code_path": "explore_code/seed_pool_000.py", "trajectory_description": "...", "code_description": "...", "type": "seed_pool", "error": {...}}
  The "type" field indicates the trajectory's purpose: "seed_pool" indicates target problem trajectories to fix, "correct_pool" indicates correct trajectories that should not be broken.
  The "error" field contains the specific prediction errors made by the INITIAL version of the code (initial /workspace/candidate_model.py) on this trajectory. Note that if the code is updated later, the actual errors might be different, but this field serves as a reference for the original problem.
  CRITICAL: You MUST read this file first to get an overview of the available trajectories, their descriptions, and the specific errors. Use this information to decide which specific trajectory files or code files you need to read in detail.

Trajectory checker:
- Script: check_traj_example.py
- Command:
  python check_traj_example.py <code_path.py> <trajectory_path>
  Example:
  python check_traj_example.py candidate_model.py traj/

Interpret checker output:
- It prints JSON metrics.
- Field meanings:
  - overall_accuracy: total_correct / total_steps over all trajectories.
  - total_correct: total number of correctly predicted frames.
  - total_steps: total number of evaluated transition steps.
  - per_trajectory_stats[*].traj_name: trajectory file identifier.
  - per_trajectory_stats[*].correct_frames: correct frame count for that trajectory.
  - per_trajectory_stats[*].total_frames: total frame count for that trajectory.
  - per_trajectory_stats[*].accuracy: per-trajectory ratio correct_frames / total_frames.
Use per-trajectory statistics to locate weak trajectories and guide further exploration or code edits.

Important coding constraints:
- Your Python file MUST define callable init_state and predict_dynamics.
- Function signatures:
  - def init_state():
  - def predict_dynamics(state, hidden_state, action):
- Return values must be tuples of length 2 exactly.
- Purpose and expected behavior:
  - init_state initializes your hidden_state and returns
    (initial_visible_state_placeholder, initial_hidden_state).
    The checker mainly uses this to obtain the initial hidden_state for rollout.
  - predict_dynamics implements the transition function:
    given current visible state, current hidden state, and action,
    return next visible state and next hidden state.
  - In checker rollout mode, each next prediction is fed into the following step,
    so early mistakes propagate. Design hidden_state updates carefully.
- visible_state/action conventions:
  - visible_state is a scene-graph-like dict (object lists + GRID_SIZE), e.g.:
    {
      "object_type_a": [{"position": {"x": 10, "y": 5}, "color": "red"}],
      "object_type_b": [{"position": {"x": 3, "y": 4}, "color": "blue"}],
      "GRID_SIZE": 20
    }
  - action is typically a dict parsed from trajectories, such as:
    {"type": "click", "x": 3, "y": 4}, {"type": "left"}, {"type": "noop"}.
- You may include any helper functions/classes.

Hard requirements:
1) Operate by actually calling the tool; do not just describe commands.
2) Read the initial model file at /workspace/candidate_model.py (if it exists).
3) When you have finished fixing the code, you MUST save the final updated code to /workspace/update_model.py.
4) You may create temporary codes and collect temporary trajectories.
5) Call collect_targeted_trajectories only when targeted evidence/trajectory pool is insufficient.
"""
_CHECK_NEED_MORE_SYSTEM_PROMPT = """\
You are an autonomous code-repair engineer working in /workspace with one tool:
- run_command_in_docker(command: str)

Goal:
- Determine if the current trajectory pool is sufficient to fix the target problem, or if more targeted exploration trajectories are needed.
- You MUST ONLY use the shell tool to READ files (e.g., using `cat`, `ls`, `grep`). Do NOT execute any code, run python scripts, or modify any files.

Environment basics:
- Deterministic GRID_SIZE x GRID_SIZE world.
- You can treat this as an MDP/POMDP-style dynamics problem: visible observations may
  be sufficient in some environments, while others require hidden_state to represent
  latent dynamics.
- Valid actions for env.step(action):
  - click x y: Click on the cell at location (x, y). For GRID_SIZE, x and y
    must each be between 0 and GRID_SIZE-1 inclusive. (Do NOT use commas, e.g., "click 3, 4" is invalid).
  - left: Press the left arrow key.
  - right: Press the right arrow key.
  - up: Press the up arrow key.
  - down: Press the down arrow key.
  - noop: Do nothing and continue to the next step.

Workspace Structure & Background:
You are operating in /workspace. You can use shell commands to view the contents of the directory as needed.
Important directories and files:
- /workspace/candidate_model.py: The current Python dynamics model code that needs to be fixed. (If this file does not exist, it means this is the initial generation).
- /workspace/explore_code/: Directory containing Python scripts used to generate the formal trajectories.
- /workspace/traj/: Directory containing the formal trajectories. Files prefixed with `seed_pool_` are target problem trajectories to fix, while `correct_pool_` indicates trajectories that the current `candidate_model` can correctly predict, which you should try your best not to break.
- /workspace/trajectory_index.jsonl: A JSONL (JSON Lines) file containing objects describing available trajectories. Each line has the following format:
  {"trajectory_path": "traj/seed_pool_000.json", "code_path": "explore_code/seed_pool_000.py", "trajectory_description": "...", "code_description": "...", "type": "seed_pool", "error": {...}}
  The "type" field indicates the trajectory's purpose: "seed_pool" indicates target problem trajectories to fix, "correct_pool" indicates correct trajectories that should not be broken.
  The "error" field contains the specific prediction errors made by the INITIAL version of the code (initial /workspace/candidate_model.py) on this trajectory. Note that if the code is updated later, the actual errors might be different, but this field serves as a reference for the original problem.
  CRITICAL: You MUST read this file first to get an overview of the available trajectories, their descriptions, and the specific errors. Use this information to decide which specific trajectory files or code files you need to read in detail.

Hard requirements:
1) Operate by actually calling the tool; do not just describe commands.
2) ONLY read files. DO NOT write, edit, or execute code.
3) Your final answer MUST be a valid JSON object indicating whether more targeted trajectories are needed.
   The JSON object must have a single boolean key: "need_more_targeted_trajectories".
   For example: {"need_more_targeted_trajectories": true}
"""




def _problem_trajectories(problem: Dict[str, Any]) -> List[TrajectoryRecord]:
    out: List[TrajectoryRecord] = []
    for sample in problem.get("problem_samples", []):
        traj = sample.get("trajectory") if isinstance(sample, dict) else None
        if isinstance(traj, dict):
            out.append(traj)
    return out


def create_problem_repair_graph(llm):
    def _cfg(state: ProblemRepairState, section: str) -> Dict[str, Any]:
        return dict((state.get("config") or {}).get(section, {}))

    def prepare_repair_runtime_node(state: ProblemRepairState) -> Dict[str, Any]:
        node_dir = create_node_dir(state.get("current_log_dir", ""), "prepare_repair_runtime")
        
        global_cfg = _cfg(state, "global_config")
        env_name = str(global_cfg.get("env_name", ""))
        
        repair_info = get_or_create_runtime_info(
            env_name=env_name,
            runtime_key=state.get("repair_runtime_key", ""),
        )
        repair_workspace_dir = repair_info.get("workspace_dir")
        
        code_sync_out = write_code_to_runtime(
            execute_run_command,
            code_text=state.get("code") or "",
            code_path="candidate_model.py",
            env_name=env_name,
            runtime_key=state.get("repair_runtime_key", ""),
            log_dir=node_dir,
        )
        
        from shared_runtime import extract_errors_by_traj_path
        
        seed_trajectories = [
            row
            for row in _problem_trajectories(state.get("target_problem", {}))
            if isinstance(row, dict)
        ]
        correct_trajectories = list(state.get("correct_trajectories", []))
        all_trajectories = list(state.get("all_trajectories", []))
        
        problem_samples = state.get("target_problem", {}).get("problem_samples", [])
        errors_by_traj_path = extract_errors_by_traj_path(problem_samples)
        
        trajectory_index = []
        if repair_workspace_dir:
            if not state.get("code", "").strip():
                # If code is empty, we sync all trajectories as correct_pool so the agent can learn from them
                trajectory_index.extend(sync_trajectories_to_workspace(
                    all_trajectories, "correct_pool", repair_workspace_dir
                ))
            else:
                trajectory_index.extend(sync_trajectories_to_workspace(
                    seed_trajectories, "seed_pool", repair_workspace_dir, errors_by_traj_path
                ))
                trajectory_index.extend(sync_trajectories_to_workspace(
                    correct_trajectories, "correct_pool", repair_workspace_dir
                ))
            
            index_path = os.path.join(repair_workspace_dir, "trajectory_index.jsonl")
            with open(index_path, "w", encoding="utf-8") as f:
                for entry in trajectory_index:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
        return {}

    def check_need_more_trajectories_node(state: ProblemRepairState) -> Dict[str, Any]:
        node_dir = create_node_dir(state.get("current_log_dir", ""), "check_need_more_trajectories")
        handler = NodeLoggingCallbackHandler(node_dir)
        
        if not state.get("code", "").strip():
            return {"need_more_targeted_trajectories": False}
            
        global_cfg = _cfg(state, "global_config")
        env_name = str(global_cfg.get("env_name", ""))
        
        tools = get_langchain_tools(
            timeout_seconds=30,
            env_name=env_name,
            runtime_key=state.get("repair_runtime_key", ""),
            log_dir=node_dir,
        )
        
        pool_size = len(_problem_trajectories(state.get("target_problem", {})))
        problem_summary = str(dict(state.get("target_problem", {})).get("summary", ""))
        
        problem_samples = state.get("target_problem", {}).get("problem_samples", [])
        filtered_errors = [s.get("error") for s in problem_samples if isinstance(s, dict) and "error" in s]
        filtered_errors_json = json.dumps(filtered_errors[:3], ensure_ascii=True, indent=2)
        
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=_CHECK_NEED_MORE_SYSTEM_PROMPT,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=10,
                    exit_behavior="end",
                )
            ],
        )
        
        user_prompt = (
            f"Target problem summary:\n{problem_summary}\n\n"
            f"Current trajectory pool size for this problem: {pool_size}\n\n"
            f"Error examples (up to 3):\n{filtered_errors_json}\n\n"
            "Full error details are saved in the `error` field of `/workspace/trajectory_index.jsonl`, please read this file.\n\n"
            "Please explore the workspace to decide if we need more trajectories. Output the JSON object."
        )
        
        final_messages: List[Any] = []
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config={"recursion_limit": 5000, "callbacks": [handler]},
            stream_mode="values",
        ):
            if isinstance(event, dict) and "messages" in event:
                final_messages = event.get("messages", []) or final_messages
                
        final_response = (
            message_content_to_text(getattr(final_messages[-1], "content", ""))
            if final_messages
            else ""
        )
        
        import re
        need_more = pool_size < 2
        try:
            match = re.search(r'\{.*"need_more_targeted_trajectories"\s*:\s*(true|false).*?\}', final_response.lower(), re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                need_more = bool(parsed.get("need_more_targeted_trajectories", need_more))
        except Exception as e:
            logger.warning(f"Failed to parse JSON from check_need_more_trajectories_node: {e}")
            
        return {"need_more_targeted_trajectories": need_more}

    def route_after_need_more(state: ProblemRepairState) -> str:
        if not state.get("target_problem"):
            return END
        if state.get("need_more_targeted_trajectories"):
            return "collect_targeted"
        return "run_shell_refine"

    def collect_targeted_node(state: ProblemRepairState) -> Dict[str, Any]:
        node_dir = create_node_dir(state.get("current_log_dir", ""), "collect_targeted")
        
        global_cfg = _cfg(state, "global_config")
        budget_cfg = _cfg(state, "budget")
        problem_repair_cfg = _cfg(state, "problem_repair")
        collect_out = call_targeted_collection_graph_from_docker_runtime(
            llm,
            env_name=str(global_cfg.get("env_name", "")),
            repair_runtime_key=state.get("repair_runtime_key", ""),
            target_problem_summary=str(dict(state.get("target_problem", {})).get("summary", "")),
            base_target_problem=state.get("target_problem"),
            code_path="candidate_model.py",
            fallback_code=state.get("code") or "",
            max_explore_steps=int(budget_cfg.get("max_explore_steps", 120)),
            collect_agent_max_turns=int(problem_repair_cfg.get("collect_agent_max_turns", 16)),
            shell_command_timeout_seconds=int(
                problem_repair_cfg.get("shell_command_timeout_seconds", 30)
            ),
            targeted_runtime_key=f"targeted-{uuid.uuid4().hex}",
            cleanup_targeted_runtime=True,
            reason="workflow_collect_targeted",
            current_log_dir=node_dir,
        )
        refreshed_target = collect_out.get("target_problem")
        if not isinstance(refreshed_target, dict):
            refreshed_target = dict(state.get("target_problem", {}))
        return {
            "target_problem": refreshed_target,
            "need_more_targeted_trajectories": False,
        }

    def run_react_refine_node(state: ProblemRepairState) -> Dict[str, Any]:
        node_dir = create_node_dir(state.get("current_log_dir", ""), "run_react_refine")
        handler = NodeLoggingCallbackHandler(node_dir)
        
        global_cfg = _cfg(state, "global_config")
        budget_cfg = _cfg(state, "budget")
        problem_repair_cfg = _cfg(state, "problem_repair")
        env_name = str(global_cfg.get("env_name", ""))
        code_path = "candidate_model.py"
        code = state.get("code") or ""
        target_problem = dict(state.get("target_problem", {}))
        
        problem_samples = target_problem.get("problem_samples", [])
        filtered_errors = [s.get("error") for s in problem_samples if isinstance(s, dict) and "error" in s]
        
        max_turns = max(
            4,
            int(problem_repair_cfg.get("shell_refine_max_turns", 20) or 20),
        )
        shell_command_timeout_seconds = max(
            10,
            int(problem_repair_cfg.get("shell_command_timeout_seconds", 30) or 30),
        )
        def _collect_targeted_trajectories(reason: str = "") -> str:
            targeted_dir = get_next_sequence_dir(node_dir, "targeted_collection")
            collect_out = call_targeted_collection_graph_from_docker_runtime(
                llm,
                env_name=env_name,
                repair_runtime_key=state.get("repair_runtime_key", ""),
                base_target_problem=target_problem,  # keep metadata, do not share mutable state
                code_path="candidate_model.py",
                fallback_code=code,
                max_explore_steps=int(budget_cfg.get("max_explore_steps", 120)),
                collect_agent_max_turns=int(
                    problem_repair_cfg.get("collect_agent_max_turns", 16)
                ),
                shell_command_timeout_seconds=int(
                    problem_repair_cfg.get("shell_command_timeout_seconds", 30)
                ),
                targeted_runtime_key=f"targeted-{uuid.uuid4().hex}",
                cleanup_targeted_runtime=True,
                reason=reason,
                current_log_dir=targeted_dir,
            )
            synced_files = collect_out.get("runtime_synced_traj_files", [])
            msg = (
                f"Successfully collected {len(synced_files)} NEW error trajectories targeting the current problem.\n"
                f"These files have been added to the 'seed_pool' and synced to runtime:\n"
                f"{', '.join(synced_files)}\n"
                f"The trajectory_index.jsonl file has been updated with these new trajectories and their corresponding errors.\n"
                f"Please note that the 'error' field in the jsonl file represents the errors made by the INITIAL version of the code. If you have updated the code, the actual errors might be different.\n"
                f"Please read these specific files to find new edge cases and clues for your repair."
            )
            return msg

        tools = get_langchain_tools(
            timeout_seconds=shell_command_timeout_seconds,
            env_name=env_name,
            runtime_key=state.get("repair_runtime_key", ""),
            log_dir=node_dir,
        )
        tools = list(tools)
        
        if code.strip():
            tools.append(
                StructuredTool.from_function(
                    func=_collect_targeted_trajectories,
                    name="collect_targeted_trajectories",
                    description=(
                        "Collect one batch of targeted trajectories for the active problem using "
                        "a dedicated collection agent, evaluate errors for that problem, and "
                        "merge results into the target problem's samples."
                    ),
                )
            )
            
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=_REFINE_SYSTEM_PROMPT,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=max_turns,
                    exit_behavior="end",
                )
            ],
        )
        
        if not code.strip():
            prompt = (
                "Write the initial dynamics code with a ReAct workflow.\n"
                f"code_path: {code_path}\n"
                "target_trajectory_dir: traj (Read trajectory_index.jsonl to find trajectories that should be predicted correctly)\n\n"
                "Please explore the workspace, read the correct trajectories, and write the initial `candidate_model.py`.\n"
                "Use run_command_in_docker to experiment and edit code. "
                "Focus on writing a model that can correctly predict the trajectories in the correct_pool."
            )
        else:
            filtered_errors_json = json.dumps(filtered_errors[:3], ensure_ascii=True, indent=2)
            prompt = (
                "Repair the current dynamics code with a ReAct workflow.\n"
                f"code_path: {code_path}\n"
                "Targeted error evidence for this problem (up to 3 examples):\n"
                f"{filtered_errors_json}\n\n"
                "Current code:\n"
                f"{code}\n\n"
                "Use run_command_in_docker to experiment and edit code. "
                "Focus on solving the active problem and improving targeted trajectory accuracy without breaking correct_pool trajectories."
            )
        
        final_messages: List[Any] = []
        for event in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": 5000, "callbacks": [handler]},
            stream_mode="values",
        ):
            if isinstance(event, dict) and "messages" in event:
                final_messages = event.get("messages", []) or final_messages
        final_response = (
            message_content_to_text(getattr(final_messages[-1], "content", ""))
            if final_messages
            else ""
        )
        latest_code = read_code_from_runtime(
            execute_run_command,
            code_path="update_model.py",
            env_name=env_name,
            runtime_key=state.get("repair_runtime_key", ""),
        )
        
        # The ReAct agent might have called `collect_targeted_trajectories` tool which writes 
        # new trajectories to the docker runtime, but doesn't update the LangGraph state directly.
        # We need to load these newly collected trajectories from the runtime and merge them 
        # into the target_problem so the next node can evaluate against them.
        runtime_out = load_all_trajectories_from_runtime(
            execute_run_command,
            get_or_create_runtime_info,
            env_name=env_name,
            runtime_key=state.get("repair_runtime_key", ""),
            timeout_seconds=shell_command_timeout_seconds,
        )
        runtime_trajectories = list(runtime_out.get("all_trajectories", []))
        
        # We simply wrap the raw trajectories into ProblemSamples without running a full 
        # error analysis, since we know these were collected specifically for this problem.
        # The exact error messages will be re-evaluated in the next node anyway.
        new_samples = [{"trajectory": t, "error": {}} for t in runtime_trajectories]
        final_target = append_problem_samples(target_problem, new_samples)

        if latest_code:
            final_code_path = Path(node_dir) / "final_code.py"
            final_code_path.write_text(latest_code, encoding="utf-8")

        return {
            "target_problem": final_target,
            "session_code_candidate": latest_code,
        }

    def evaluate_and_update_problem_status_node(state: ProblemRepairState) -> Dict[str, Any]:
        node_dir = create_node_dir(state.get("current_log_dir", ""), "evaluate_and_update_problem_status")
        
        target_trajs = [
            row
            for row in _problem_trajectories(state.get("target_problem", {}))
            if isinstance(row, dict)
        ]
        correct_trajectories = list(state.get("correct_trajectories", []))
        all_eval_trajs = target_trajs + correct_trajectories
        
        is_initial_generation = not state.get("code", "").strip()
        if is_initial_generation:
            all_eval_trajs = list(state.get("all_trajectories", []))
            target_trajs = []
            
        target_eval = run_error_analysis(
            llm,
            code=state.get("session_code_candidate", state.get("code", "")),
            trajectories=all_eval_trajs,
            problems=[],
            target_problem=state.get("target_problem"),
        )
        target_problem_errors = target_eval.get("target_problem_errors", [])
        
        # We also need to check if there are any errors on correct_trajectories.
        # run_error_analysis returns all error_samples before filtering.
        error_samples = target_eval.get("error_samples", [])
        regression_errors = [
            err for err in error_samples 
            if isinstance(err.get("traj_idx"), int) and err.get("traj_idx") >= len(target_trajs)
        ]
        
        # Problem P is considered solved when no remaining errors belong to P,
        # AND no regressions were introduced in correct_trajectories.
        perfect = len(target_problem_errors) == 0 and len(regression_errors) == 0

        problem = dict(state.get("target_problem", {}))
        updated = problem
        if problem:
            updated = dict(problem)
            updated["tried"] = True
            if perfect:
                updated["solved"] = True
            current_samples = list(problem.get("problem_samples", []))
            # We only want to update the problem samples with errors from target_trajs,
            # not from correct_trajectories.
            target_traj_errors = [
                err for err in target_problem_errors 
                if isinstance(err.get("traj_idx"), int) and err.get("traj_idx") < len(target_trajs)
            ]
            refreshed_samples = build_problem_samples_from_errors(
                target_trajs,
                target_traj_errors,
            )
            updated = append_problem_samples(
                {**problem, "problem_samples": current_samples},
                refreshed_samples,
            )
            
        out_state = {
            "target_problem_fixed": perfect,
            "target_problem": updated if updated else state.get("target_problem", {}),
        }
        
        is_initial_generation = not state.get("code", "").strip()
        if perfect or is_initial_generation:
            out_state["code"] = state.get("session_code_candidate", state.get("code", ""))
            
        return out_state

    workflow = StateGraph(ProblemRepairState)
    workflow.add_node("prepare_repair_runtime", prepare_repair_runtime_node)
    workflow.add_node("check_need_more_trajectories", check_need_more_trajectories_node)
    workflow.add_node("collect_targeted", collect_targeted_node)
    workflow.add_node("run_react_refine", run_react_refine_node)
    workflow.add_node("evaluate_and_update_problem_status", evaluate_and_update_problem_status_node)
    workflow.set_entry_point("prepare_repair_runtime")
    workflow.add_edge("prepare_repair_runtime", "check_need_more_trajectories")
    workflow.add_conditional_edges(
        "check_need_more_trajectories",
        route_after_need_more,
        {"collect_targeted": "collect_targeted", "run_shell_refine": "run_react_refine", END: END},
    )
    workflow.add_edge("collect_targeted", "run_react_refine")
    workflow.add_edge("run_react_refine", "evaluate_and_update_problem_status")
    workflow.add_edge("evaluate_and_update_problem_status", END)
    return workflow.compile()


def run_problem_repair_subgraph(llm, state: WorkflowState) -> Dict[str, Any]:
    parent_log_dir = state.get("current_log_dir", "")
    subgraph_dir = create_subgraph_dir(parent_log_dir, "problem_repair") if parent_log_dir else ""
    
    config = dict(state.get("config") or {})
    global_cfg = dict(config.get("global_config", {}))
    code = state.get("code", "")
    target_problem = dict(state.get("target_problem") or {})
    target_problem_id = str(target_problem.get("problem_id", "")).strip()
    if not target_problem_id and code.strip():
        return {
            "target_problem": None,
            "code": code,
            "all_trajectories": state.get("all_trajectories", []),
            "correct_trajectories": state.get("correct_trajectories", []),
            "problems": state.get("problems"),
            "target_problem_fixed": False,
            "session_code_candidate": code,
        }

    graph = create_problem_repair_graph(llm)
    repair_runtime_key = f"repair-{target_problem_id or 'initial'}-{uuid.uuid4().hex}"
    get_or_create_runtime_info(
        env_name=str(global_cfg.get("env_name", "")),
        runtime_key=repair_runtime_key,
    )
    out: Dict[str, Any] = {}
    try:
        out = graph.invoke(
            {
                "target_problem": target_problem,
                "config": config,
                "code": code,
                "session_code_candidate": code,
                "correct_trajectories": state.get("correct_trajectories", []),
                "all_trajectories": state.get("all_trajectories", []),
                "target_problem_fixed": state.get("target_problem_fixed", False),
                "need_more_targeted_trajectories": state.get(
                    "need_more_targeted_trajectories", False
                ),
                "repair_runtime_key": repair_runtime_key,
                "targeted_runtime_key": "",
                "current_log_dir": subgraph_dir,
            }
        )
    finally:
        close_runtime(
            env_name=str(global_cfg.get("env_name", "")),
            runtime_key=repair_runtime_key,
        )
    updated_target = dict(out.get("target_problem", target_problem) or target_problem)
    repaired_ok = bool(out.get("target_problem_fixed", False))
    next_code = str(out.get("code", code) or "")
    return {
        "code": next_code,
        "all_trajectories": state.get("all_trajectories", []),
        "correct_trajectories": state.get("correct_trajectories", []),
        # Re-analyze only when code changed after a successful repair, or if it was the initial generation.
        "problems": None if (repaired_ok or not code.strip()) else state.get("problems"),
        "target_problem": updated_target,
        "target_problem_fixed": repaired_ok,
        "session_code_candidate": out.get("session_code_candidate", code),
    }

