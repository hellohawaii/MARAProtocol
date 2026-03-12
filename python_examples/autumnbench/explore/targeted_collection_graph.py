"""
Targeted trajectory collection subgraph.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from error_analysis_graph import run_error_analysis
from shared_runtime import (
    ProblemItem,
    ProblemSample,
    WorkflowConfig,
    WorkflowState,
    TrajectoryRecord,
    append_problem_samples,
    build_problem_samples_from_errors,
)
from explore_by_code.env_wrapper import execute_run_command, get_or_create_runtime_info
from runtime_utils import (
    load_all_trajectories_from_runtime as load_all_trajectories_from_runtime_util,
    read_code_from_runtime as read_code_from_runtime_util,
    write_trajectories_to_runtime as write_trajectories_to_runtime_util,
)
from shell_collect_runtime import run_collect_agent, TARGETED_COLLECT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class TargetedPlan(BaseModel):
    hypothesis: str = Field(default="")
    actions: List[str] = Field(default_factory=list)


class TargetedCollectionState(TypedDict):
    # Shared config
    config: WorkflowConfig

    # Active collection context for current target problem.
    code: str
    target_problem: ProblemItem
    collect_hint: str
    current_hypothesis: str

    # Output data produced within this subgraph.
    targeted_new_trajectories: List[TrajectoryRecord]

    # Runtime routing keys and info.
    targeted_runtime_key: Optional[str]




def create_targeted_collection_graph(llm):
    def _cfg(state: TargetedCollectionState, section: str) -> Dict[str, Any]:
        return dict((state.get("config") or {}).get(section, {}))

    # def propose_strategy_node(state: TargetedCollectionState) -> Dict[str, Any]:
    #     forced_hint = state.get("collect_hint")
    #     if forced_hint and forced_hint.strip():
    #         return {"current_hypothesis": "forced-hint", "collect_hint": forced_hint.strip()}
    #         
    #     target_problem = dict(state.get("target_problem", {}))
    #     summary = str(target_problem.get("summary", ""))
    #     
    #     problem_samples = target_problem.get("problem_samples", [])
    #     filtered_errors = [s.get("error") for s in problem_samples if isinstance(s, dict) and "error" in s]
    #     filtered_errors_json = json.dumps(filtered_errors[:3], ensure_ascii=True, indent=2)
    #     
    #     prompt = (
    #         "Design a trigger strategy around the following problem, and output the action sequence in JSON format."
    #         f"\nProblem: {summary}\n"
    #         f"Error examples (up to 3):\n{filtered_errors_json}\n"
    #         "Format: {hypothesis: str, actions: [str,...]}, number of actions should be 8~20."
    #     )
    #     try:
    #         structured = llm.with_structured_output(TargetedPlan)
    #         plan: TargetedPlan = structured.invoke(prompt)
    #         actions = plan.actions or ["noop"]
    #         hypothesis = plan.hypothesis
    #     except Exception:
    #         actions = ["noop", "left", "right"]
    #         hypothesis = "fallback-plan"
    #         
    #     action_hint = (
    #         f"target hypothesis: {hypothesis}\n"
    #         f"preferred actions: {', '.join(str(a) for a in actions)}"
    #     )
    #     
    #     return {"current_hypothesis": hypothesis, "collect_hint": action_hint}

    def collect_trajectory_node(state: TargetedCollectionState) -> Dict[str, Any]:
        global_cfg = _cfg(state, "global_config")
        budget_cfg = _cfg(state, "budget")
        targeted_cfg = _cfg(state, "targeted_collection")
        
        target_problem = dict(state.get("target_problem", {}))
        summary = str(target_problem.get("summary", ""))
        problem_samples = target_problem.get("problem_samples", [])
        filtered_errors = [s.get("error") for s in problem_samples if isinstance(s, dict) and "error" in s]
        filtered_errors_json = json.dumps(filtered_errors[:3], ensure_ascii=True, indent=2)

        objective = (
            "There is no code yet. Please collect initial trajectories based on the current hypothesis.\n"
            if not state.get("code") else
            "The current code makes prediction errors on certain trajectories.\n"
            f"These errors likely belong to the following category/summary:\n{summary}\n\n"
            f"Error examples (up to 3):\n{filtered_errors_json}\n\n"
            "Full error details are saved in the `error` field of `/workspace/trajectory_index.jsonl`, please read this file to understand the specific errors you need to reproduce.\n\n"
            "Your task is to collect more trajectories that trigger similar errors based on the current hypothesis, "
            "providing rich evidence and information for subsequent reasoning and code repair.\n"
            "Focus on reproducing the currently selected issue."
        )
        
        collect_out = run_collect_agent(
            llm,
            env_name=str(global_cfg.get("env_name", "")),
            max_explore_steps=int(budget_cfg.get("max_explore_steps", 120)),
            objective=str(state.get("collect_hint", "")).strip() + f"\nObjective: {objective}",
            code=state.get("code"),
            max_turns=int(targeted_cfg.get("collect_agent_max_turns", 16)),
            timeout_seconds=int(targeted_cfg.get("shell_command_timeout_seconds", 30)),
            runtime_key=state.get("targeted_runtime_key"),
            cleanup_runtime=bool(targeted_cfg.get("cleanup_targeted_runtime", False)),
            system_prompt=TARGETED_COLLECT_SYSTEM_PROMPT,
        )
        
        trajectories = list(collect_out.get("trajectories", []))
        if not trajectories:
            raise RuntimeError("targeted collect agent returned empty trajectories")
            
        runtime_info = collect_out.get("runtime_info", {})
        hypothesis = state.get("current_hypothesis", "")
        
        return {
            "targeted_new_trajectories": trajectories,
            "targeted_runtime_key": collect_out.get("runtime_key", state.get("targeted_runtime_key")),
        }

    def evaluate_and_update_problem_node(state: TargetedCollectionState) -> Dict[str, Any]:
        target_problem = dict(state.get("target_problem", {}))
        new_trajectories = list(state.get("targeted_new_trajectories", []))
        
        if not state.get("code", "").strip():
            new_samples = build_problem_samples_from_errors(
                new_trajectories,
                [],
            )
            updated_problem = append_problem_samples(target_problem, new_samples)
            return {"target_problem": updated_problem}
            
        analysis_out = run_error_analysis(
            llm,
            code=(state.get("code") or "").strip(),
            trajectories=new_trajectories,
            problems=[target_problem] if target_problem else [],
            target_problem=target_problem,
        )
        filtered_errors = list(analysis_out.get("target_problem_errors", []))
        
        new_samples = build_problem_samples_from_errors(
            new_trajectories,
            filtered_errors,
        )
        updated_problem = append_problem_samples(target_problem, new_samples)
        
        return {
            "target_problem": updated_problem,
        }

    workflow = StateGraph(TargetedCollectionState)
    # workflow.add_node("propose_strategy", propose_strategy_node)
    workflow.add_node("collect_trajectory", collect_trajectory_node)
    workflow.add_node("evaluate_and_update_problem", evaluate_and_update_problem_node)
    # workflow.set_entry_point("propose_strategy")
    # workflow.add_edge("propose_strategy", "collect_trajectory")
    workflow.set_entry_point("collect_trajectory")
    workflow.add_edge("collect_trajectory", "evaluate_and_update_problem")
    workflow.add_edge("evaluate_and_update_problem", END)
    return workflow.compile()


def run_targeted_collection_subgraph(llm, state: WorkflowState) -> Dict[str, Any]:
    target_problem = dict(state.get("target_problem") or {})
    target_problem["problem_samples"] = list(target_problem.get("problem_samples", []))
    
    # Initialize targeted collection state
    initial_state: TargetedCollectionState = {
        "config": state.get("config", {}),
        "code": state.get("code", ""),
        "target_problem": target_problem, # type: ignore[arg-type]
        "collect_hint": state.get("collect_hint", ""),
        "current_hypothesis": "",
        "targeted_new_trajectories": [],
        "targeted_runtime_key": state.get("targeted_runtime_key"),
    }
    
    workflow = create_targeted_collection_graph(llm)
    out_state = workflow.invoke(initial_state)
    
    updated_target = out_state.get("target_problem", target_problem)
    return {
        "target_problem": updated_target,
        "problems": [updated_target] if updated_target else [],
        "targeted_runtime_key": out_state.get("targeted_runtime_key"),
        "targeted_new_trajectories": out_state.get("targeted_new_trajectories", []),
        "collect_hint": out_state.get("collect_hint", ""),
    }


def call_targeted_collection_graph_from_docker_runtime(
    llm: Any,
    *,
    env_name: str,
    repair_runtime_key: str,
    base_target_problem: Optional[ProblemItem] = None,
    code_path: str = "candidate_model.py",
    fallback_code: str = "",
    max_explore_steps: int = 120,
    collect_hint: Optional[str] = None,
    collect_agent_max_turns: int = 16,
    shell_command_timeout_seconds: int = 30,
    targeted_runtime_key: Optional[str] = None,
    cleanup_targeted_runtime: bool = True,
    reason: str = "",
) -> Dict[str, Any]:
    target_problem = dict(base_target_problem or {})
    target_problem["problem_samples"] = list(target_problem.get("problem_samples", []))
    latest_code = read_code_from_runtime_util(
        execute_run_command,
        env_name=env_name,
        runtime_key=repair_runtime_key,
        code_path=code_path,
        fallback_code=fallback_code,
    )
    
    state: WorkflowState = { # type: ignore
        "config": {
            "global_config": {"env_name": env_name},
            "budget": {"max_explore_steps": max_explore_steps},
            "targeted_collection": {
                "collect_agent_max_turns": collect_agent_max_turns,
                "shell_command_timeout_seconds": shell_command_timeout_seconds,
                "cleanup_targeted_runtime": cleanup_targeted_runtime,
            }
        },
        "code": latest_code,
        "target_problem": target_problem,
        "collect_hint": collect_hint or "",
        "targeted_runtime_key": targeted_runtime_key,
    }
    
    out = run_targeted_collection_subgraph(llm, state)
    
    repair_info = get_or_create_runtime_info(
        env_name=env_name,
        runtime_key=repair_runtime_key,
    )
    repair_workspace_dir = repair_info.get("workspace_dir")
    
    synced_files = []
    if repair_workspace_dir:
        from shared_runtime import sync_trajectories_to_workspace, extract_errors_by_traj_path
        
        updated_target_problem = dict(out.get("target_problem", {}))
        problem_samples = updated_target_problem.get("problem_samples", [])
        errors_by_traj_path = extract_errors_by_traj_path(problem_samples)
                        
        new_index_entries = sync_trajectories_to_workspace(
            list(out.get("targeted_new_trajectories", [])),
            "seed_pool",
            repair_workspace_dir,
            errors_by_traj_path
        )
        
        if new_index_entries:
            import os, json
            index_path = os.path.join(repair_workspace_dir, "trajectory_index.jsonl")
            with open(index_path, "a", encoding="utf-8") as f:
                for entry in new_index_entries:
                    entry["trajectory_description"] = "Newly collected targeted trajectory for the active problem."
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            synced_files = [e["trajectory_path"] for e in new_index_entries]
    
    runtime_loaded = load_all_trajectories_from_runtime_util(
        execute_run_command,
        get_or_create_runtime_info,
        env_name=env_name,
        runtime_key=repair_runtime_key,
        timeout_seconds=shell_command_timeout_seconds,
    )
    all_trajectories = list(runtime_loaded.get("all_trajectories", []))
    
    if not latest_code.strip():
        filtered_errors = []
    else:
        analysis_out = run_error_analysis(
            llm,
            code=latest_code,
            trajectories=all_trajectories,
            problems=[target_problem] if target_problem else [],
            target_problem=target_problem,
        )
        filtered_errors = list(analysis_out.get("target_problem_errors", []))
    
    new_samples = build_problem_samples_from_errors(
        all_trajectories,
        filtered_errors,
    )
    refreshed_target = append_problem_samples(target_problem, new_samples)
    refreshed_samples = list(refreshed_target.get("problem_samples", []))
    
    return {
        **out,
        "target_problem": refreshed_target,
        "code_from_runtime": latest_code,
        "runtime_synced_traj_files": synced_files,
        "runtime_sync_output": "Synced via shared_runtime helper",
        "runtime_all_traj_files": runtime_loaded.get("all_traj_files", []),
        "runtime_load_output": runtime_loaded.get("load_output", ""),
    }

