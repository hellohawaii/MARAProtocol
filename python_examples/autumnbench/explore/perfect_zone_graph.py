"""
Perfect-zone exploration subgraph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TypedDict, Optional

from langgraph.graph import END, StateGraph
from langchain.agents.middleware import ModelCallLimitMiddleware

from code_refine_graph import execute_code_on_trajectories
from shared_runtime import WorkflowConfig, WorkflowState, TrajectoryRecord
from shell_collect_runtime import run_collect_agent, DEFAULT_DOCKERFILE_PATH
from explore_by_code.env_wrapper import get_langchain_tools, get_or_create_runtime_info
from runtime_utils import message_content_to_text
from langchain.agents import create_agent
from pathlib import Path

from log_manager import create_subgraph_dir, create_node_dir, NodeLoggingCallbackHandler

logger = logging.getLogger(__name__)


class PerfectZoneState(TypedDict):
    config: WorkflowConfig

    code: str # Current code to be validated.
    new_trajectories: List[TrajectoryRecord] # Newly collected trajectories from current perfect-zone loop.
    all_trajectories: List[TrajectoryRecord] # All trajectories collected so far.

    has_unexplained_frames: bool # Whether current code still has unexplained frames on recent probes.
    explored_round_count: int # Number of completed perfect-zone exploration rounds.
    stop_exploration_early: bool # Early-stop flag decided by the question proposer.

    current_log_dir: str # Logging directory passed down the graph


def create_perfect_zone_graph(llm):
    def _cfg(state: PerfectZoneState, section: str) -> Dict[str, Any]:
        return dict((state.get("config") or {}).get(section, {}))

    def decide_stop_node(state: PerfectZoneState) -> Dict[str, Any]:
        node_dir = create_node_dir(state.get("current_log_dir", ""), "decide_stop")
        handler = NodeLoggingCallbackHandler(node_dir)
        
        code = (state.get("code") or "").strip()
        if not code:
            logger.info("PerfectZone decide_stop: No code available, forcing CONTINUE.")
            return {
                "stop_exploration_early": False,
            }

        round_idx = int(state.get("explored_round_count", 0))
        perfect_cfg = _cfg(state, "perfect_zone")
        max_rounds = int(perfect_cfg.get("max_perfect_zone_rounds", 20))
        if round_idx >= max_rounds:
            return {
                "stop_exploration_early": True,
            }
            
        global_cfg = _cfg(state, "global_config")
        env_name = str(global_cfg.get("env_name", ""))
        timeout_seconds = int(perfect_cfg.get("shell_command_timeout_seconds", 30))
        runtime_key = "perfect-zone"
        
        get_or_create_runtime_info(
            dockerfile_path=DEFAULT_DOCKERFILE_PATH,
            env_name=env_name,
            runtime_key=runtime_key,
        )
        tools = get_langchain_tools(
            timeout_seconds=timeout_seconds,
            dockerfile_path=DEFAULT_DOCKERFILE_PATH,
            env_name=env_name,
            runtime_key=runtime_key,
            log_dir=node_dir,
        )
        system_prompt = """\
You are an autonomous exploration decision maker operating inside Docker.
You have ONE tool:
- run_command_in_docker(command: str)
  Execute shell commands in a persistent container workspace at /workspace.
  Reuse shell state and files across commands.

Your mission is to evaluate the current state of environment exploration by reading the exploration log.
You are operating in /workspace. You can use shell commands to view the contents of the directory as needed.

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

Important directories and files:
- /workspace/explore_code/: Directory for your formal exploration Python scripts.
- /workspace/traj/: Directory where env.save_trajectory() automatically saves formal trajectories.
- /workspace/exploration_log.jsonl: A JSONL (JSON Lines) file containing logs of formal explorations. Each line is a JSON object with: "code_path", "trajectory_path", "code_description", and "trajectory_description".

Please use the shell tool to read the those files, especially the exploration log file. Determine if the exploration is sufficient to fully understand the environment's dynamics.
If the exploration is sufficient and covers enough edge cases, you MUST output exactly `STOP`. If more exploration is needed, you MUST output exactly `CONTINUE`.\
"""
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=10,
                    exit_behavior="end",
                )
            ],
        )
        user_prompt = "Please read the log file /workspace/exploration_log.jsonl and decide whether to stop exploration (STOP or CONTINUE)."
        
        final_messages = []
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config={"recursion_limit": 5000, "callbacks": [handler]},
            stream_mode="values",
        ):
            if isinstance(event, dict) and "messages" in event:
                final_messages = event.get("messages", []) or final_messages
                
        final_text = ""
        if final_messages:
            final_text = message_content_to_text(getattr(final_messages[-1], "content", ""))
            
        should_stop = "STOP" in final_text.upper()
        logger.info("PerfectZone decide_stop: %s", should_stop)
        return {
            "stop_exploration_early": should_stop,
        }

    def route_after_decide(state: PerfectZoneState) -> str:
        if state.get("stop_exploration_early", False):
            return END
        return "collect_trajectory"

    def collect_trajectory_node(state: PerfectZoneState) -> Dict[str, Any]:
        node_dir = create_node_dir(state.get("current_log_dir", ""), "collect_trajectory")
        handler = NodeLoggingCallbackHandler(node_dir)
        
        global_cfg = _cfg(state, "global_config")
        budget_cfg = _cfg(state, "budget")
        perfect_cfg = _cfg(state, "perfect_zone")
        
        code = (state.get("code") or "").strip()
        if not code:
            objective = (
                "You have absolutely no knowledge of the environment yet. "
                "Your task is to explore the environment and collect some initial trajectories to understand the basic dynamics. "
                "You don't need to generate too many trajectories at once, as this exploration process will iterate multiple times."
            )
        else:
            objective = (
                "The current code explains the already collected trajectories, but there may be unexplored states or rules in the environment. "
                "Your task is to explore new trajectories, novel states, or edge cases that are not yet covered by the current code."
            )
        
        collect_out = run_collect_agent(
            llm,
            env_name=str(global_cfg.get("env_name", "")),
            objective=objective,
            code=state.get("code"),
            max_explore_steps=int(budget_cfg.get("max_explore_steps", 120)),
            max_turns=int(perfect_cfg.get("collect_agent_max_turns", 100)),
            timeout_seconds=int(perfect_cfg.get("shell_command_timeout_seconds", 30)),
            runtime_key="perfect-zone",
            log_file_path="/workspace/exploration_log.jsonl",
            log_dir=node_dir,
            callbacks=[handler],
        )
        trajectories = list(collect_out.get("trajectories", []))
        if not trajectories:
            raise RuntimeError("collect agent returned empty trajectories")
        latest = list(state.get("new_trajectories", []))
        latest.extend(trajectories)
        library = list(state.get("all_trajectories", []))
        library.extend(trajectories)
        return {
            "new_trajectories": latest,
            "all_trajectories": library,
            "explored_round_count": int(state.get("explored_round_count", 0)) + 1,
        }

    def validate_with_code_node(state: PerfectZoneState) -> Dict[str, Any]:
        node_dir = create_node_dir(state.get("current_log_dir", ""), "validate_with_code")
        
        code = (state.get("code") or "").strip()
        latest = list(state.get("new_trajectories", []))
        if not code or not latest:
            return {
                "has_unexplained_frames": True,
            }
        result = execute_code_on_trajectories(code, latest)
        if not result.get("success"):
            return {
                "has_unexplained_frames": True,
            }
        rows = result.get("results", [])
        total_correct = sum(r.get("correct", 0) for r in rows)
        total_frames = sum(r.get("total", 0) for r in rows)
        acc = total_correct / total_frames if total_frames > 0 else 1.0
        return {
            "has_unexplained_frames": acc < 1.0,
        }

    def route_after_validate(state: PerfectZoneState) -> str:
        if state.get("has_unexplained_frames", False):
            return END
        # No unexplained errors for current probe; continue exploration.
        return "decide_stop"

    workflow = StateGraph(PerfectZoneState)
    workflow.add_node("decide_stop", decide_stop_node)
    workflow.add_node("collect_trajectory", collect_trajectory_node)
    workflow.add_node("validate_with_code", validate_with_code_node)
    workflow.set_entry_point("decide_stop")
    workflow.add_conditional_edges(
        "decide_stop",
        route_after_decide,
        {"collect_trajectory": "collect_trajectory", END: END},
    )
    workflow.add_edge("collect_trajectory", "validate_with_code")
    workflow.add_conditional_edges(
        "validate_with_code",
        route_after_validate,
        {"decide_stop": "decide_stop", END: END},
    )
    return workflow.compile()


def run_perfect_zone_subgraph(llm, state: WorkflowState) -> Dict[str, Any]:
    parent_log_dir = state.get("current_log_dir", "")
    subgraph_dir = create_subgraph_dir(parent_log_dir, "perfect_zone") if parent_log_dir else ""
    
    graph = create_perfect_zone_graph(llm)
    config = dict(state.get("config") or {})
    code = state.get("code", "")
    all_trajectories = list(state.get("all_trajectories", []))
    input_state: Dict[str, Any] = {
        "code": code,
        "config": config,
        "new_trajectories": [],
        "all_trajectories": all_trajectories,
        "has_unexplained_frames": False,
        "explored_round_count": 0,
        "stop_exploration_early": False,
        "current_log_dir": subgraph_dir,
    }
    out = graph.invoke(input_state)
    
    new_trajectories = out.get("new_trajectories", [])
    if subgraph_dir and new_trajectories:
        from explore_by_code.traj_visualization_utils import save_trajectory_visualization
        from runtime_utils import trajectory_to_saved_payload
        import os
        
        traj_dir = os.path.join(subgraph_dir, "traj")
        os.makedirs(traj_dir, exist_ok=True)
        
        for idx, traj in enumerate(new_trajectories):
            payload = trajectory_to_saved_payload(traj)
            if payload:
                out_path = Path(traj_dir) / f"trajectory_{idx:03d}.json"
                save_trajectory_visualization(payload, out_path, run_id=None, log_dir=subgraph_dir)

    return {
        "code": code,
        "new_trajectories": new_trajectories,
        "all_trajectories": out.get("all_trajectories", all_trajectories),
        "correct_trajectories": state.get("correct_trajectories", []),
        "has_unexplained_frames": bool(out.get("has_unexplained_frames", False)),
        # New probes can make previous problem analysis stale.
        "problems": None,
    }

