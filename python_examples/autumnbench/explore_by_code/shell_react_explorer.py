"""Minimal ReAct explorer + code synthesis agent.

This script intentionally does not encode a workflow for exploration/refinement.
The LLM gets environment tools plus lightweight HCI phase-declaration tools and
decides how to explore, write Python files, run check_traj_example.py, and when
to stop.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

_FILE_DIR = Path(__file__).resolve().parent
_AUTUMNBENCH_DIR = _FILE_DIR.parent
_MARA_ROOT = _AUTUMNBENCH_DIR.parents[1]
for _p in [str(_AUTUMNBENCH_DIR), str(_MARA_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_utils import get_llm  # noqa: E402

from env_wrapper import (  # noqa: E402
    get_env_tools,
    get_env_tools_for_runtime,
    get_or_create_runtime_info,
)
from hci_tools import get_ask_human_tool, get_dashboard_tools, get_finish_tool, get_hci_tools, FINISH_TOOL_NAME  # noqa: E402
from log_utils import _CHECK_TRAJ_PATTERN  # noqa: E402
from shell_react_prompt import (  # noqa: E402
    SHELL_REACT_HUMAN_COLLAB_SYSTEM_PROMPT,
    SHELL_REACT_MODEL_ORCHESTRATED_SYSTEM_PROMPT,
    SHELL_REACT_PLANNING_HUMAN_COLLAB_SYSTEM_PROMPT,
    SHELL_REACT_PLANNING_MODEL_ORCHESTRATED_SYSTEM_PROMPT,
    SHELL_REACT_PLANNING_SYSTEM_PROMPT,
    SHELL_REACT_SYSTEM_PROMPT,
    SHELL_REACT_VARIANT_BATCH_EVAL_SYSTEM_PROMPT,
    build_initial_user_prompt,
    build_variant_batch_eval_user_prompt,
)

DEFAULT_DOCKERFILE_PATH = str((_FILE_DIR / "Dockerfile.tool").resolve())

_EXAMPLE_BENCHMARK_DIR = (_AUTUMNBENCH_DIR / "example_benchmark").resolve()

from planning_utils import (  # noqa: E402
    load_color_dict,
    goal_to_color_grid,
    color_grid_to_scene_graph,
    mask_to_positions,
)
from variant_env_utils import load_goal_and_mask  # noqa: E402


def _load_planning_goal(env_name: str):
    """Load planning goal/mask and convert to scene_graph format.

    Returns (goal_scene_graph_json, mask_scene_graph_json) as JSON strings,
    or (None, None) if no planning data exists.
    """
    data_dir, result = load_goal_and_mask(_EXAMPLE_BENCHMARK_DIR, env_name)
    if result is None:
        return None, None

    raw_goal, raw_mask = result
    color_dict = load_color_dict(data_dir)
    goal_color = goal_to_color_grid(raw_goal, color_dict)
    goal_sg = color_grid_to_scene_graph(goal_color)

    grid_size = len(goal_color)
    all_ones = all(raw_mask[r][c] == 1 for r in range(grid_size) for c in range(grid_size))
    if all_ones:
        mask_repr = "FULL_GRID"
    else:
        mask_repr = mask_to_positions(raw_mask)

    return (
        json.dumps(goal_sg, indent=2),
        json.dumps(mask_repr, indent=2) if not isinstance(mask_repr, str) else f'"{mask_repr}"',
    )


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable(vars(obj))
        except Exception:
            pass
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _message_to_jsonable(msg: Any) -> Dict[str, Any]:
    return {
        "type": getattr(msg, "type", msg.__class__.__name__),
        "content": _message_content_to_text(getattr(msg, "content", "")),
        "tool_calls": _to_jsonable(getattr(msg, "tool_calls", None)),
        "additional_kwargs": _to_jsonable(getattr(msg, "additional_kwargs", {})),
        "response_metadata": _to_jsonable(getattr(msg, "response_metadata", {})),
    }


class _JsonTraceCallback(BaseCallbackHandler):
    """Persist low-level per-call LLM inputs/outputs as JSON files."""

    def __init__(self, detail_log_dir: Path):
        self.detail_log_dir = detail_log_dir

    def _write_event(self, event: str, run_id: Any, payload: Dict[str, Any]) -> None:
        ts = int(time.time() * 1000)
        rid = str(run_id).replace("-", "")
        out_path = self.detail_log_dir / "llm_traces" / f"{ts}_{rid}_{event}.json"
        _write_json(out_path, payload)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        run_id: Any,
        **kwargs: Any,
    ) -> Any:
        self._write_event(
            "chat_start",
            run_id,
            {
                "serialized": serialized,
                "messages": [[_message_to_jsonable(m) for m in batch] for batch in messages],
            },
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        run_id: Any,
        **kwargs: Any,
    ) -> Any:
        self._write_event(
            "llm_start",
            run_id,
            {
                "serialized": serialized,
                "prompts": prompts,
            },
        )

    def on_llm_end(self, response: Any, run_id: Any, **kwargs: Any) -> Any:
        generations: List[Any] = []
        for group in getattr(response, "generations", []) or []:
            row: List[Any] = []
            for gen in group:
                text = getattr(gen, "text", None)
                message = getattr(gen, "message", None)
                row.append(
                    {
                        "text": text if text is not None else "",
                        "message": _message_to_jsonable(message) if message is not None else None,
                    }
                )
            generations.append(row)
        self._write_event(
            "llm_end",
            run_id,
            {
                "generations": generations,
                "llm_output": getattr(response, "llm_output", {}),
            },
        )


def _messages_to_jsonable(messages: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for msg in messages:
        rows.append(_message_to_jsonable(msg))
    return rows


def _is_terminal_ai_without_tool_calls(messages: List[Any]) -> bool:
    if not messages:
        return False
    last_msg = messages[-1]
    msg_type = (getattr(last_msg, "type", "") or last_msg.__class__.__name__).lower()
    if msg_type not in ("ai", "aimessage"):
        return False
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    return len(tool_calls) == 0


def _last_tool_messages_contain(messages: List[Any], tool_name: str) -> bool:
    """Check whether any trailing ToolMessage was produced by the given tool."""
    for msg in reversed(messages):
        msg_type = (getattr(msg, "type", "") or msg.__class__.__name__).lower()
        if msg_type in ("tool", "toolmessage"):
            if getattr(msg, "name", None) == tool_name:
                return True
        else:
            break  # stop at the first non-ToolMessage
    return False


def _tool_call_args(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize tool call args to a dict for downstream state tracking."""
    args = tool_call.get("args", {})
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _select_system_prompt(task_mode: str, collaborative: bool, orchestrator: str) -> str:
    if task_mode == "planning":
        if orchestrator == "model":
            return SHELL_REACT_PLANNING_MODEL_ORCHESTRATED_SYSTEM_PROMPT
        if collaborative:
            return SHELL_REACT_PLANNING_HUMAN_COLLAB_SYSTEM_PROMPT
        return SHELL_REACT_PLANNING_SYSTEM_PROMPT
    if orchestrator == "model":
        return SHELL_REACT_MODEL_ORCHESTRATED_SYSTEM_PROMPT
    if collaborative:
        return SHELL_REACT_HUMAN_COLLAB_SYSTEM_PROMPT
    return SHELL_REACT_SYSTEM_PROMPT


async def arun_variant_batch_eval_agent(
    env_name: str,
    *,
    user_instruction: str,
    initial_state: Dict[str, Any],
    runtime,
    llm_model: str = "openai/gpt-5.4",
    max_turns: int = 120,
    timeout_seconds: int = 30,
    yield_state_callback=None,
) -> Dict[str, Any]:
    llm = get_llm(model=llm_model)
    runtime.configure_environment(env_name=env_name, task_mode="planning")
    runtime.install_batch_eval_env_client()
    runtime.set_client_control_mode("restricted")
    env_tools = get_env_tools_for_runtime(runtime, timeout_seconds=timeout_seconds)

    agent = create_agent(
        model=llm,
        tools=env_tools,
        system_prompt=SHELL_REACT_VARIANT_BATCH_EVAL_SYSTEM_PROMPT,
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=max_turns,
                exit_behavior="end",
            )
        ],
    )

    goal_sg, mask_sg = _load_planning_goal(env_name)
    user_prompt = build_variant_batch_eval_user_prompt(
        env_name=env_name,
        user_instruction=user_instruction,
        initial_state=json.dumps(initial_state, indent=2),
        goal_scene_graph=goal_sg or "null",
        mask_scene_graph=mask_sg or "null",
    )

    transcript_path = (
        _FILE_DIR / "logs" / runtime.run_id / f"variant_batch_eval_{env_name}_transcript.json"
    ).resolve()
    detail_log_dir = transcript_path.with_suffix("") / "details"
    trace_cb = _JsonTraceCallback(detail_log_dir)

    final_messages: List[Any] = []
    stream_event_count = 0
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"recursion_limit": 1000, "callbacks": [trace_cb]},
        stream_mode="values",
    ):
        stream_event_count += 1
        if isinstance(event, dict) and "messages" in event:
            final_messages = event.get("messages", []) or final_messages
            _write_json(
                detail_log_dir / "stream_events" / f"{stream_event_count:04d}.json",
                {
                    "stream_event_idx": stream_event_count,
                    "num_messages": len(final_messages),
                    "last_message": (
                        _message_to_jsonable(final_messages[-1]) if final_messages else None
                    ),
                    "messages": _messages_to_jsonable(final_messages),
                },
            )

            if yield_state_callback and final_messages:
                last_msg = final_messages[-1]
                msg_type = getattr(last_msg, "type", "") or last_msg.__class__.__name__
                if msg_type in ("tool", "ToolMessage"):
                    try:
                        trajectory_payload = runtime.backend_get_trajectory()
                    except Exception:
                        trajectory_payload = None
                    if trajectory_payload:
                        await yield_state_callback(trajectory_payload)

    final_response = ""
    if final_messages:
        final_response = _message_content_to_text(getattr(final_messages[-1], "content", ""))

    result: Dict[str, Any] = {
        "ok": True,
        "env_name": env_name,
        "llm_model": llm_model,
        "max_turns": max_turns,
        "timeout_seconds": timeout_seconds,
        "num_messages": len(final_messages),
        "final_response": final_response,
        "runtime_info": runtime.get_runtime_info(),
        "stream_event_count": stream_event_count,
    }

    payload = {
        **result,
        "messages": _messages_to_jsonable(final_messages),
    }
    _write_json(transcript_path, payload)
    result["transcript_path"] = str(transcript_path)
    result["detail_log_dir"] = str(detail_log_dir)

    return result


def run_shell_react_agent(
    env_name: str,
    *,
    llm_model: str = "google/gemini-3.1-pro-preview",
    max_turns: int = 1000,
    timeout_seconds: int = 30,
    docker_image: Optional[str] = None,
    dockerfile_path: Optional[str] = DEFAULT_DOCKERFILE_PATH,
    docker_build_context: Optional[str] = None,
    env_api_base_url: str = "http://host.docker.internal:8002",
    collaborative: bool = False,
    task_mode: str = "explore",
) -> Dict[str, Any]:
    llm = get_llm(model=llm_model)
    runtime_info = get_or_create_runtime_info(
        docker_image=docker_image,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
        env_name=env_name,
        task_mode=task_mode,
    )
    env_tools = get_env_tools(
        docker_image=docker_image,
        timeout_seconds=timeout_seconds,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
        env_name=env_name,
        task_mode=task_mode,
    )
    hci_tools = get_hci_tools(task_mode=task_mode)
    tools = env_tools + hci_tools
    if collaborative:
        tools = tools + get_dashboard_tools()
    system_prompt = _select_system_prompt(task_mode, collaborative, "developer")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        # system_prompt = "You are a chat bot with shell tool.",
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=max_turns,
                exit_behavior="end",
            )
        ],
    )

    goal_sg, mask_sg = (None, None)
    if task_mode == "planning":
        goal_sg, mask_sg = _load_planning_goal(env_name)
    user_prompt = build_initial_user_prompt(
        env_name,
        collaborative=collaborative,
        task_mode=task_mode,
        goal_scene_graph=goal_sg or "",
        mask_scene_graph=mask_sg or "",
    )

    # Persist logs under the same run_id used by llm_workspace_runs.
    run_id = runtime_info.get("run_id")
    if run_id:
        transcript_path = (
            _FILE_DIR / "logs" / run_id / f"shell_react_{env_name}_transcript.json"
        ).resolve()
    else:
        ts = int(time.time() * 1000)
        transcript_path = (
            _FILE_DIR / "logs" / f"shell_react_{env_name}_{ts}_transcript.json"
        ).resolve()
    detail_log_dir = transcript_path.with_suffix("") / "details"
    trace_cb = _JsonTraceCallback(detail_log_dir)

    workspace_dir = Path(runtime_info.get("workspace_dir", ""))
    current_code_file = None
    current_code_content = None
    current_traj_files = set()
    current_traj_contents = {}
    current_world_model_description = ""
    current_plan = ""
    current_phase = ""

    final_messages: List[Any] = []
    stream_event_count = 0
    for event in agent.stream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"recursion_limit": 1000, "callbacks": [trace_cb]},
        stream_mode="values",
    ):
        stream_event_count += 1
        if isinstance(event, dict) and "messages" in event:
            final_messages = event.get("messages", []) or final_messages
            
            if final_messages:
                last_msg = final_messages[-1]
                
                # Track which code file is being validated.
                # NOTE: At tool_call time, the shell command has not executed yet,
                # so reading file content here can be stale.
                if getattr(last_msg, "tool_calls", None):
                    for tool_call in last_msg.tool_calls:
                        tool_name = tool_call.get("name")
                        tool_args = _tool_call_args(tool_call)
                        if tool_name == "run_command_in_docker":
                            command = tool_args.get("command", "")
                            match = _CHECK_TRAJ_PATTERN.search(command)
                            if match:
                                current_code_file = match.group("code").strip("'\"")
                                print(f"\n[State Update] Validating code file: {current_code_file}")
                        elif tool_name == "update_world_model_description":
                            current_world_model_description = str(
                                tool_args.get("description", "")
                            )
                        elif tool_name == "update_plan":
                            current_plan = str(tool_args.get("plan", ""))
                        elif tool_name and tool_name.startswith("declare_phase_"):
                            current_phase = tool_name[len("declare_phase_") :]

                # Capture new trajectories and content
                msg_type = getattr(last_msg, "type", "") or last_msg.__class__.__name__
                if msg_type in ("tool", "ToolMessage"):
                    content = _message_content_to_text(getattr(last_msg, "content", ""))
                    matches = re.findall(r'"saved_path"\s*:\s*"([^"]+)"', content)
                    for saved_path in matches:
                        if saved_path not in current_traj_files:
                            current_traj_files.add(saved_path)
                            if workspace_dir:
                                abs_traj_path = workspace_dir / saved_path
                                if abs_traj_path.is_file():
                                    current_traj_contents[saved_path] = json.loads(abs_traj_path.read_text(encoding="utf-8"))
                            print(f"\n[State Update] New trajectory saved: {saved_path}")

                    # Refresh code content after the shell tool command has executed.
                    if workspace_dir and current_code_file:
                        abs_code_path = workspace_dir / current_code_file
                        if abs_code_path.is_file():
                            current_code_content = abs_code_path.read_text(encoding="utf-8")
                            print(f"\n[State Update] Code content updated: {current_code_content}")

            _write_json(
                detail_log_dir / "stream_events" / f"{stream_event_count:04d}.json",
                {
                    "stream_event_idx": stream_event_count,
                    "num_messages": len(final_messages),
                    "last_message": (
                        _message_to_jsonable(final_messages[-1]) if final_messages else None
                    ),
                    "messages": _messages_to_jsonable(final_messages),
                },
            )

    final_response = ""
    if final_messages:
        final_response = _message_content_to_text(getattr(final_messages[-1], "content", ""))

    result: Dict[str, Any] = {
        "ok": True,
        "env_name": env_name,
        "llm_model": llm_model,
        "max_turns": max_turns,
        "timeout_seconds": timeout_seconds,
        "num_messages": len(final_messages),
        "final_response": final_response,
        "runtime_info": runtime_info,
        "stream_event_count": stream_event_count,
    }

    payload = {
        **result,
        "messages": _messages_to_jsonable(final_messages),
    }
    _write_json(transcript_path, payload)
    result["transcript_path"] = str(transcript_path)
    result["detail_log_dir"] = str(detail_log_dir)

    return result



async def arun_shell_react_agent(
    env_name: str,
    *,
    llm_model: str = "google/gemini-3.1-pro-preview",
    max_turns: int = 1000,
    timeout_seconds: int = 30,
    docker_image: Optional[str] = None,
    dockerfile_path: Optional[str] = DEFAULT_DOCKERFILE_PATH,
    docker_build_context: Optional[str] = None,
    env_api_base_url: str = "http://host.docker.internal:8002",
    yield_state_callback=None,
    wait_for_user_input_callback=None,
    collaborative: bool = False,
    orchestrator: str = "developer",
    task_mode: str = "explore",
    pause_gate: Optional[asyncio.Event] = None,
    notify_paused_callback=None,
    input_queue: Optional[asyncio.Queue] = None,
) -> Dict[str, Any]:
    terminal_waiting_message = "AI thinks exploration is complete. Waiting for input!"
    default_waiting_message = "Waiting for input!"

    async def _wait_for_user_input(waiting_message: str) -> Any:
        if not wait_for_user_input_callback:
            return None
        try:
            return await wait_for_user_input_callback(waiting_message)
        except TypeError:
            # Backward compatibility for callbacks that take no arguments.
            return await wait_for_user_input_callback()

    llm = get_llm(model=llm_model)
    runtime_info = get_or_create_runtime_info(
        docker_image=docker_image,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
        env_name=env_name,
        task_mode=task_mode,
    )
    env_tools = get_env_tools(
        docker_image=docker_image,
        timeout_seconds=timeout_seconds,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
        env_name=env_name,
        task_mode=task_mode,
    )
    hci_tools = get_hci_tools(task_mode=task_mode)
    tools = env_tools + hci_tools
    if collaborative or orchestrator == "model":
        tools = tools + get_dashboard_tools()

    if orchestrator == "model":
        if wait_for_user_input_callback:
            ask_human_tool = get_ask_human_tool(wait_for_user_input_callback)
            tools = tools + [ask_human_tool]
        finish_tool = get_finish_tool()
        tools = tools + [finish_tool]
    elif orchestrator == "user":
        finish_tool = get_finish_tool()
        tools = tools + [finish_tool]

    system_prompt = _select_system_prompt(task_mode, collaborative, orchestrator)

    checkpointer = MemorySaver()

    use_interrupt = (orchestrator == "user") or (
        orchestrator == "developer" and wait_for_user_input_callback
    ) or (orchestrator == "model")

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=max_turns,
                exit_behavior="end",
            )
        ],
        checkpointer=checkpointer,
        interrupt_after=["tools"] if use_interrupt else None,
    )

    goal_sg, mask_sg = (None, None)
    if task_mode == "planning":
        goal_sg, mask_sg = _load_planning_goal(env_name)
    user_prompt = build_initial_user_prompt(
        env_name,
        collaborative=collaborative,
        orchestrator=orchestrator,
        task_mode=task_mode,
        goal_scene_graph=goal_sg or "",
        mask_scene_graph=mask_sg or "",
    )

    # Persist logs under the same run_id used by llm_workspace_runs.
    run_id = runtime_info.get("run_id")
    if run_id:
        transcript_path = (
            _FILE_DIR / "logs" / run_id / f"shell_react_{env_name}_transcript.json"
        ).resolve()
    else:
        ts = int(time.time() * 1000)
        transcript_path = (
            _FILE_DIR / "logs" / f"shell_react_{env_name}_{ts}_transcript.json"
        ).resolve()
    detail_log_dir = transcript_path.with_suffix("") / "details"
    trace_cb = _JsonTraceCallback(detail_log_dir)

    workspace_dir = Path(runtime_info.get("workspace_dir", ""))
    current_code_file = None
    current_code_content = None
    current_traj_files = set()
    current_traj_contents = {}
    current_world_model_description = ""
    current_plan = ""
    current_phase = ""

    final_messages: List[Any] = []
    stream_event_count = 0

    config = {"configurable": {"thread_id": run_id or str(ts)}, "recursion_limit": 1000, "callbacks": [trace_cb]}
    
    input_state = {"messages": [{"role": "user", "content": user_prompt}]}

    async def _wait_for_human_approval_to_end(paused_state: str) -> Optional[str]:
        if pause_gate and notify_paused_callback:
            await notify_paused_callback(paused_state)
            pause_gate.clear()
            await pause_gate.wait()
            if input_queue:
                try:
                    return input_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return None
            return None
        if wait_for_user_input_callback:
            user_input = await _wait_for_user_input(terminal_waiting_message)
            return user_input if user_input and user_input.strip() else None
        return None
    
    while True:
        state_changed = False
        
        async for event in agent.astream(
            input_state,
            config=config,
            stream_mode="values",
        ):
            input_state = None  # only needed for the first run or after update_state
            stream_event_count += 1
            if isinstance(event, dict) and "messages" in event:
                final_messages = event.get("messages", []) or final_messages
                
                if final_messages:
                    last_msg = final_messages[-1]
                    
                    # Track which code file is being validated.
                    # NOTE: At tool_call time, the shell command has not executed yet,
                    # so reading file content here can be stale.
                    if getattr(last_msg, "tool_calls", None):
                        for tool_call in last_msg.tool_calls:
                            tool_name = tool_call.get("name")
                            tool_args = _tool_call_args(tool_call)
                            if tool_name == "run_command_in_docker":
                                command = tool_args.get("command", "")
                                match = _CHECK_TRAJ_PATTERN.search(command)
                                if match:
                                    current_code_file = match.group("code").strip("'\"")
                                    print(f"\n[State Update] Validating code file: {current_code_file}")
                            elif tool_name == "update_world_model_description":
                                new_description = str(tool_args.get("description", ""))
                                if new_description != current_world_model_description:
                                    current_world_model_description = new_description
                                    # state_changed = True
                            elif tool_name == "update_plan":
                                new_plan = str(tool_args.get("plan", ""))
                                if new_plan != current_plan:
                                    current_plan = new_plan
                                    # state_changed = True
                            elif tool_name and tool_name.startswith("declare_phase_"):
                                new_phase = tool_name[len("declare_phase_") :]
                                if new_phase != current_phase:
                                    current_phase = new_phase

                    # Capture new trajectories and content
                    msg_type = getattr(last_msg, "type", "") or last_msg.__class__.__name__
                    if msg_type in ("tool", "ToolMessage"):
                        content = _message_content_to_text(getattr(last_msg, "content", ""))
                        matches = re.findall(r'"saved_path"\s*:\s*"([^"]+)"', content)
                        for saved_path in matches:
                            if saved_path not in current_traj_files:
                                current_traj_files.add(saved_path)
                                if workspace_dir:
                                    abs_traj_path = workspace_dir / saved_path
                                    if abs_traj_path.is_file():
                                        current_traj_contents[saved_path] = json.loads(abs_traj_path.read_text(encoding="utf-8"))
                                        state_changed = True
                                print(f"\n[State Update] New trajectory saved: {saved_path}")

                        # Refresh code content after the shell tool command has executed.
                        if workspace_dir and current_code_file:
                            abs_code_path = workspace_dir / current_code_file
                            if abs_code_path.is_file():
                                new_content = abs_code_path.read_text(encoding="utf-8")
                                if new_content != current_code_content:
                                    current_code_content = new_content
                                    state_changed = True

                _write_json(
                    detail_log_dir / "stream_events" / f"{stream_event_count:04d}.json",
                    {
                        "stream_event_idx": stream_event_count,
                        "num_messages": len(final_messages),
                        "last_message": (
                            _message_to_jsonable(final_messages[-1]) if final_messages else None
                        ),
                        "messages": _messages_to_jsonable(final_messages),
                    },
                )
                
                if yield_state_callback:
                    await yield_state_callback({
                        "current_code_file": current_code_file,
                        "current_code_content": current_code_content,
                        "current_traj_files": list(current_traj_files),
                        "current_traj_contents": current_traj_contents,
                        "current_world_model_description": current_world_model_description,
                        "current_plan": current_plan,
                        "current_phase": current_phase,
                        "messages": _messages_to_jsonable(final_messages[-5:]), # last 5 messages
                    })

        # graph paused or finished
        state = await agent.aget_state(config)

        if not state.next:
            # Agent finished (terminal AI message without tool calls, or turn limit)
            if _is_terminal_ai_without_tool_calls(final_messages):
                if orchestrator in ("model", "user"):
                    # Model/user stopped without finish_task — force continuation.
                    await agent.aupdate_state(config, {"messages": []}, as_node="tools")
                    input_state = None
                    continue
                else:
                    # developer mode (or user mode without pause_gate): ask for input
                    if wait_for_user_input_callback:
                        user_input = await _wait_for_user_input(terminal_waiting_message)
                        if user_input and user_input.strip():
                            input_state = {"messages": [HumanMessage(content=user_input)]}
                            continue
                    break
            break

        # state.next is non-empty: agent wants to continue (interrupt_after checkpoint).
        if orchestrator == "model":
            if _last_tool_messages_contain(final_messages, FINISH_TOOL_NAME):
                break  # exit — model explicitly signaled completion
            input_state = None
            continue  # resume react loop
        elif orchestrator == "user":
            if _last_tool_messages_contain(final_messages, FINISH_TOOL_NAME):
                human_msg = await _wait_for_human_approval_to_end("paused")
                if human_msg and human_msg.strip():
                    await agent.aupdate_state(
                        config, {"messages": [HumanMessage(content=human_msg)]}
                    )
                    input_state = None
                    continue
                break
            if pause_gate and not pause_gate.is_set():
                if notify_paused_callback:
                    await notify_paused_callback("paused")
                await pause_gate.wait()
                human_msg = None
                if input_queue:
                    try:
                        human_msg = input_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                if human_msg and human_msg.strip():
                    await agent.aupdate_state(
                        config, {"messages": [HumanMessage(content=human_msg)]}
                    )
            input_state = None
            continue
        else:
            # developer: ask for input when state changed
            if state_changed and wait_for_user_input_callback:
                user_input = await _wait_for_user_input(default_waiting_message)
                if user_input and user_input.strip():
                    await agent.aupdate_state(config, {"messages": [HumanMessage(content=user_input)]})
                
    final_response = ""
    if final_messages:
        final_response = _message_content_to_text(getattr(final_messages[-1], "content", ""))

    result: Dict[str, Any] = {
        "ok": True,
        "env_name": env_name,
        "llm_model": llm_model,
        "max_turns": max_turns,
        "timeout_seconds": timeout_seconds,
        "num_messages": len(final_messages),
        "final_response": final_response,
        "runtime_info": runtime_info,
        "stream_event_count": stream_event_count,
    }

    payload = {
        **result,
        "messages": _messages_to_jsonable(final_messages),
    }
    _write_json(transcript_path, payload)
    result["transcript_path"] = str(transcript_path)
    result["detail_log_dir"] = str(detail_log_dir)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal ReAct agent with a single shell tool for environment "
            "exploration and Python code synthesis."
        )
    )
    parser.add_argument("env_name", default="7XF97", help="Environment id, e.g. 7XF97")
    parser.add_argument(
        "--llm-model",
        default="google/gemini-3-pro-preview",
        help="OpenRouter model id.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1000,
        help="Safety recursion limit for the agent loop.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Timeout for each run_command_in_docker command.",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Optional base Docker image. Used only when --dockerfile-path is not set.",
    )
    parser.add_argument(
        "--dockerfile-path",
        default=DEFAULT_DOCKERFILE_PATH,
        help="Dockerfile path used to build tool image (defaults to Dockerfile.tool).",
    )
    parser.add_argument(
        "--docker-build-context",
        default=None,
        help="Optional docker build context path.",
    )
    parser.add_argument(
        "--env-api-base-url",
        default="http://host.docker.internal:8002",
        help="Base URL for env API server reachable from container.",
    )
    parser.add_argument(
        "--task-mode",
        default="explore",
        choices=["explore", "planning"],
        help="Task mode: 'explore' for free exploration, 'planning' for goal-directed.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    result = run_shell_react_agent(
        env_name=args.env_name,
        llm_model=args.llm_model,
        max_turns=args.max_turns,
        timeout_seconds=args.timeout_seconds,
        docker_image=args.docker_image,
        dockerfile_path=args.dockerfile_path,
        docker_build_context=args.docker_build_context,
        env_api_base_url=args.env_api_base_url,
        task_mode=args.task_mode,
    )

    print(f"Run finished. messages={result.get('num_messages')}")
    if result.get("transcript_path"):
        print(f"Transcript: {result['transcript_path']}")
    if result.get("detail_log_dir"):
        print(f"Detail logs: {result['detail_log_dir']}")
    print("Final response:")
    print(result.get("final_response", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
