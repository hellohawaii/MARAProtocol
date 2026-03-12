"""
Top-level workflow graph for the new explore pipeline.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from global_controller import choose_latest_strategy
from perfect_zone_graph import run_perfect_zone_subgraph
from problem_repair_graph import run_problem_repair_subgraph
from shared_runtime import WorkflowState

logger = logging.getLogger(__name__)

_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parents[2]


def _build_experiment_dirs(env_name: str) -> Dict[str, str]:
    root = os.path.join(str(_REPO_ROOT), "MARAProtocol", "experiments")
    os.makedirs(root, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"{env_name}_workflow_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    return {"experiments_root_dir": root, "experiment_run_dir": run_dir}


def create_workflow_graph(llm):
    def perfect_zone_node(state: WorkflowState) -> Dict[str, Any]:
        out = run_perfect_zone_subgraph(llm, state)
        return out

    def route_after_perfect_zone(state: WorkflowState) -> str:
        if state.get("has_unexplained_frames", False):
            return "global_controller"
        return END

    def global_controller_node(state: WorkflowState) -> Dict[str, Any]:
        return choose_latest_strategy(llm, state)

    def route_after_global_controller(state: WorkflowState) -> str:
        if state.get("target_problem"):
            return "repair_problem"
        if not state.get("problems"):
            return "perfect_zone"
        if state.get("should_terminate_workflow", False):
            return END
        return END

    def repair_problem_node(state: WorkflowState) -> Dict[str, Any]:
        return run_problem_repair_subgraph(llm, state)

    workflow = StateGraph(WorkflowState)
    workflow.add_node("perfect_zone", perfect_zone_node)
    workflow.add_node("global_controller", global_controller_node)
    workflow.add_node("repair_problem", repair_problem_node)

    workflow.set_entry_point("perfect_zone")
    workflow.add_conditional_edges("perfect_zone", route_after_perfect_zone, {"global_controller": "global_controller", END: END})
    workflow.add_conditional_edges(
        "global_controller",
        route_after_global_controller,
        {"repair_problem": "repair_problem", "perfect_zone": "perfect_zone", END: END},
    )
    workflow.add_edge("repair_problem", "global_controller")
    return workflow.compile()


def build_initial_state(
    env_name: str,
    data_dir: str,
    llm_model: str,
    max_explore_steps: int,
    use_obfuscation: bool = False,
) -> Dict[str, Any]:
    dirs = _build_experiment_dirs(env_name)
    return {
        "code": "",
        "config": {
            "global_config": {
                "env_name": env_name,
                "data_dir": data_dir,
                "llm_model": llm_model,
                "use_obfuscation": use_obfuscation,
            },
            "budget": {
                "max_explore_steps": max_explore_steps,
            },
            "perfect_zone": {
                "collect_agent_max_turns": 16,
                "shell_command_timeout_seconds": 30,
                "max_perfect_zone_rounds": 20,
            },
            "problem_repair": {
                "collect_agent_max_turns": 16,
                "shell_command_timeout_seconds": 30,
                "shell_refine_max_turns": 20,
                "cleanup_targeted_runtime": False,
            },
            "targeted_collection": {
                "collect_agent_max_turns": 16,
                "shell_command_timeout_seconds": 30,
                "cleanup_targeted_runtime": False,
            },
        },
        "all_trajectories": [],
        "correct_trajectories": [],
        "new_trajectories": [],
        "error_samples": [],
        "problems": None,
        "target_problem": None,
        "target_problem_fixed": False,
        "should_terminate_workflow": False,
        "need_more_targeted_trajectories": False,
        "has_unexplained_frames": False,
        "coding_can_explain": False,
        "repair_runtime_key": None,
        "session_code_candidate": None,
        "experiment_run_dir": dirs["experiment_run_dir"],
        "current_loop_dir": "",
        "explore_log_dir": "",
        "refine_log_dir": "",
    }

