"""
Shared shell-tool collection runtime for explore graphs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

_FILE_DIR = Path(__file__).resolve().parent
_AUTUMNBENCH_DIR = _FILE_DIR.parent
_REPO_ROOT = _AUTUMNBENCH_DIR.parents[2]
for _p in [str(_AUTUMNBENCH_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_by_code.env_wrapper import (  # noqa: E402
    close_runtime,
    execute_run_command,
    get_langchain_tools,
    get_or_create_runtime_info,
)
from runtime_utils import (  # noqa: E402
    load_all_trajectories_from_runtime,
    message_content_to_text,
)

DEFAULT_DOCKERFILE_PATH = str((_AUTUMNBENCH_DIR / "explore_by_code" / "Dockerfile.tool").resolve())

PERFECT_ZONE_COLLECT_SYSTEM_PROMPT = """\
You are an autonomous environment data collector operating inside Docker.
You have ONE tool:
- run_command_in_docker(command: str)
  Execute shell commands in a persistent container workspace at /workspace.
  Reuse shell state and files across commands.

Your mission is to explore a deterministic interactive grid environment and collect targeted trajectories based on the provided objectives and current code models. Note that any provided code is a draft and may not perfectly reflect the true environment dynamics; use it as a reference to guide your exploration (e.g., to find unmodeled behaviors or reproduce specific errors).

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
- /workspace/explore_code/: Directory for your formal exploration Python scripts.
- /workspace/traj/: Directory where env.save_trajectory() automatically saves formal trajectories.
- /workspace/tmp_explore_code/: Directory for temporary experimental scripts.
- /workspace/tmp_traj/: Directory for temporary experimental trajectories.
- /workspace/exploration_log.jsonl: A JSONL (JSON Lines) file you MUST maintain to log your formal explorations. Each line is a JSON object with: "code_path", "trajectory_path", "code_description", and "trajectory_description". If the file does not exist or is empty, it means no previous exploration has been done. You can safely append to it (e.g., using `echo '{"code_path": ...}' >> /workspace/exploration_log.jsonl`).

Temporary Experiments:
Before formally generating trajectories, you can conduct temporary experiments to ensure your code produces valuable trajectories.
- Write temporary scripts in /workspace/tmp_explore_code/tmp_trial.py.
- Save temporary trajectories using env.save_trajectory(filename, dir="tmp_traj").
- The tmp_explore_code and tmp_traj directories will be automatically deleted in subsequent processes, so you can freely add or overwrite files here.

Formal Exploration Constraints:
Your main goal is to collect valuable trajectories and save them to the traj folder.
1) Formal exploration must be done via Python scripts placed in /workspace/explore_code/ with unique names (e.g., explore_1.py).
2) The script MUST import RemoteEnvWrapper from env_api_client and call env.reset(), env.step(action), and env.save_trajectory(filename).
3) Calling env.save_trajectory(filename) without the dir argument automatically saves the trajectory to /workspace/traj.
4) You MUST manage the exploration log file (/workspace/exploration_log.jsonl). It is in JSONL format. For each formal exploration, you must append a single JSON object line to this file containing:
   - "code_path": Python code path
   - "trajectory_path": Saved trajectory path
   - "code_description": Brief description of the code
   - "trajectory_description": Brief description of observations from the trajectory
5) Do not output markdown commands; run actual shell commands through the tool.
"""

TARGETED_COLLECT_SYSTEM_PROMPT = """\
You are an autonomous environment data collector operating inside Docker.
You have ONE tool:
- run_command_in_docker(command: str)
  Execute shell commands in a persistent container workspace at /workspace.
  Reuse shell state and files across commands.

Your mission is to explore a deterministic interactive grid environment and collect targeted trajectories based on the provided objectives and current code models. Note that any provided code is a draft and may not perfectly reflect the true environment dynamics; use it as a reference to guide your exploration (e.g., to find unmodeled behaviors or reproduce specific errors).

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
- /workspace/explore_code/: Directory for your formal exploration Python scripts.
- /workspace/traj/: Directory containing the formal trajectories. Files prefixed with `seed_pool_` are target problem trajectories to fix, while `correct_pool_` indicates trajectories that the current `candidate_model` can correctly predict, which you should try your best not to break.
- /workspace/trajectory_index.jsonl: A JSONL (JSON Lines) file containing objects describing available trajectories. Each line has the following format:
  {"trajectory_path": "traj/seed_pool_000.json", "code_path": "explore_code/seed_pool_000.py", "trajectory_description": "...", "code_description": "...", "type": "seed_pool", "error": {...}}
  The "type" field indicates the trajectory's purpose: "seed_pool" indicates target problem trajectories to fix, "correct_pool" indicates correct trajectories that should not be broken.
  If code is provided, the "error" field contains the specific prediction errors made by the INITIAL version of the code on this trajectory. Note that if the code is updated later, the actual errors might be different, but this field serves as a reference for the original problem.
- /workspace/tmp_explore_code/: Directory for temporary experimental scripts.
- /workspace/tmp_traj/: Directory for temporary experimental trajectories.
- /workspace/exploration_log.jsonl: A JSONL (JSON Lines) file you MUST maintain to log your formal explorations. Each line is a JSON object with: "code_path", "trajectory_path", "code_description", and "trajectory_description". If the file does not exist or is empty, it means no previous exploration has been done. You can safely append to it (e.g., using `echo '{"code_path": ...}' >> /workspace/exploration_log.jsonl`).

Temporary Experiments:
Before formally generating trajectories, you can conduct temporary experiments to ensure your code produces valuable trajectories.
- Write temporary scripts in /workspace/tmp_explore_code/tmp_trial.py.
- Save temporary trajectories using env.save_trajectory(filename, dir="tmp_traj").
- The tmp_explore_code and tmp_traj directories will be automatically deleted in subsequent processes, so you can freely add or overwrite files here.

Formal Exploration Constraints:
Your main goal is to collect valuable trajectories and save them to the traj folder.
1) Formal exploration must be done via Python scripts placed in /workspace/explore_code/ with unique names (e.g., explore_1.py).
2) The script MUST import RemoteEnvWrapper from env_api_client and call env.reset(), env.step(action), and env.save_trajectory(filename).
3) Calling env.save_trajectory(filename) without the dir argument automatically saves the trajectory to /workspace/traj.
4) You MUST manage the exploration log file (/workspace/exploration_log.jsonl). It is in JSONL format. For each formal exploration, you must append a single JSON object line to this file containing:
   - "code_path": Python code path
   - "trajectory_path": Saved trajectory path
   - "code_description": Brief description of the code
   - "trajectory_description": Brief description of observations from the trajectory
5) Do not output markdown commands; run actual shell commands through the tool.
"""


def run_collect_agent(
    llm: Any,
    *,
    env_name: str,
    max_explore_steps: int,
    objective: Optional[str] = None,
    code: Optional[str] = None,
    max_turns: int = 16,
    timeout_seconds: int = 30,
    dockerfile_path: Optional[str] = DEFAULT_DOCKERFILE_PATH,
    docker_image: Optional[str] = None,
    docker_build_context: Optional[str] = None,
    env_api_base_url: str = "http://host.docker.internal:8000",
    runtime_key: Optional[str] = None,
    cleanup_runtime: bool = False,
    log_file_path: str = "/workspace/exploration_log.jsonl",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    actual_runtime_key = runtime_key
    try:
        runtime_info = get_or_create_runtime_info(
            docker_image=docker_image,
            dockerfile_path=dockerfile_path,
            docker_build_context=docker_build_context,
            env_api_base_url=env_api_base_url,
            env_name=env_name,
            runtime_key=actual_runtime_key,
        )
        tools = get_langchain_tools(
            docker_image=docker_image,
            timeout_seconds=timeout_seconds,
            dockerfile_path=dockerfile_path,
            docker_build_context=docker_build_context,
            env_api_base_url=env_api_base_url,
            env_name=env_name,
            runtime_key=actual_runtime_key,
        )
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt or PERFECT_ZONE_COLLECT_SYSTEM_PROMPT,
            middleware=[
                ModelCallLimitMiddleware(
                    run_limit=max_turns,
                    exit_behavior="end",
                )
            ],
        )

        prompt_objective = ""
        if objective and str(objective).strip():
            prompt_objective = "\nObjective:\n" + str(objective).strip()
        
        code_context = ""
        if code and str(code).strip():
            code_context = f"\nCurrent Code Draft:\n```python\n{str(code).strip()}\n```\n"

        user_prompt = (
            "Please start the exploration and collect at least 1 new trajectory.\n"
            f"Constraint: Max steps per trajectory in your script must be <= {int(max_explore_steps)}.\n"
            "Remember to follow the formal exploration constraints from the system prompt.\n"
            + code_context
            + prompt_objective
        )

        final_messages: List[Any] = []
        for event in agent.stream(
            {"messages": [{"role": "user", "content": user_prompt}]},
            config={"recursion_limit": 5000},
            stream_mode="values",
        ):
            if isinstance(event, dict) and "messages" in event:
                final_messages = event.get("messages", []) or final_messages

        load_out = load_all_trajectories_from_runtime(
            execute_run_command_fn=execute_run_command,
            get_or_create_runtime_info_fn=get_or_create_runtime_info,
            env_name=env_name,
            runtime_key=actual_runtime_key,
            timeout_seconds=min(timeout_seconds, 20),
        )

        normalized = load_out.get("all_trajectories", [])
        if not normalized:
            load_output = str(load_out.get("load_output", ""))
            raise RuntimeError(f"collect agent produced no valid trajectories. list output: {load_output[:400]}")

        from shared_runtime import attach_metadata_to_trajectories
        workspace_dir = runtime_info.get("workspace_dir", "")
        attach_metadata_to_trajectories(
            execute_run_command_fn=execute_run_command,
            env_name=env_name,
            runtime_key=actual_runtime_key,
            log_file_path=log_file_path,
            normalized_trajectories=normalized,
            workspace_dir=workspace_dir,
        )

        return {
            "trajectories": normalized,
            "runtime_info": runtime_info,
            "runtime_key": actual_runtime_key,
            "num_agent_messages": len(final_messages),
            "final_agent_message": (
                message_content_to_text(getattr(final_messages[-1], "content", ""))
                if final_messages
                else ""
            ),
        }
    finally:
        if cleanup_runtime:
            close_runtime(
                docker_image=docker_image,
                dockerfile_path=dockerfile_path,
                docker_build_context=docker_build_context,
                env_api_base_url=env_api_base_url,
                env_name=env_name,
                runtime_key=actual_runtime_key,
            )

