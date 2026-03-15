"""Minimal shell-tool ReAct explorer + code synthesis agent.

This script intentionally does not encode a workflow for exploration/refinement.
The LLM gets one shell tool and decides how to explore, write Python files,
run check_traj_example.py, and when to stop.
"""

import argparse
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
    get_langchain_tools,
    get_or_create_runtime_info,
)
from log_utils import _CHECK_TRAJ_PATTERN  # noqa: E402
from shell_react_prompt import (  # noqa: E402
    SHELL_REACT_HUMAN_COLLAB_SYSTEM_PROMPT,
    SHELL_REACT_SYSTEM_PROMPT,
    build_initial_user_prompt,
)

DEFAULT_DOCKERFILE_PATH = str((_FILE_DIR / "Dockerfile.tool").resolve())


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
) -> Dict[str, Any]:
    llm = get_llm(model=llm_model)
    runtime_info = get_or_create_runtime_info(
        docker_image=docker_image,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
        env_name=env_name,
    )
    tools = get_langchain_tools(
        docker_image=docker_image,
        timeout_seconds=timeout_seconds,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
        env_name=env_name,
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            SHELL_REACT_HUMAN_COLLAB_SYSTEM_PROMPT
            if collaborative
            else SHELL_REACT_SYSTEM_PROMPT
        ),
        # system_prompt = "You are a chat bot with shell tool.",
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=max_turns,
                exit_behavior="end",
            )
        ],
    )

    user_prompt = build_initial_user_prompt(env_name, collaborative=collaborative)
    # user_prompt = "Hi!"

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
                        if tool_call.get("name") == "run_command_in_docker":
                            command = tool_call.get("args", {}).get("command", "")
                            match = _CHECK_TRAJ_PATTERN.search(command)
                            if match:
                                current_code_file = match.group("code").strip("'\"")
                                print(f"\n[State Update] Validating code file: {current_code_file}")

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
) -> Dict[str, Any]:
    llm = get_llm(model=llm_model)
    runtime_info = get_or_create_runtime_info(
        docker_image=docker_image,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
        env_name=env_name,
    )
    tools = get_langchain_tools(
        docker_image=docker_image,
        timeout_seconds=timeout_seconds,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
        env_name=env_name,
    )
    
    checkpointer = MemorySaver()
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            SHELL_REACT_HUMAN_COLLAB_SYSTEM_PROMPT
            if collaborative
            else SHELL_REACT_SYSTEM_PROMPT
        ),
        # system_prompt = "You are a chat bot with shell tool.",
        middleware=[
            ModelCallLimitMiddleware(
                run_limit=max_turns,
                exit_behavior="end",
            )
        ],
        checkpointer=checkpointer,
        interrupt_after=["tools"] if wait_for_user_input_callback else None,
    )

    user_prompt = build_initial_user_prompt(env_name, collaborative=collaborative)
    # user_prompt = "List current files"

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

    final_messages: List[Any] = []
    stream_event_count = 0
    
    config = {"configurable": {"thread_id": run_id or str(ts)}, "recursion_limit": 1000, "callbacks": [trace_cb]}
    
    input_state = {"messages": [{"role": "user", "content": user_prompt}]}
    
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
                            if tool_call.get("name") == "run_command_in_docker":
                                command = tool_call.get("args", {}).get("command", "")
                                match = _CHECK_TRAJ_PATTERN.search(command)
                                if match:
                                    current_code_file = match.group("code").strip("'\"")
                                    print(f"\n[State Update] Validating code file: {current_code_file}")

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
                        "messages": _messages_to_jsonable(final_messages[-5:]), # last 5 messages
                    })

        # graph paused or finished
        state = await agent.aget_state(config)
        if not state.next:
            break # finished
            
        if state_changed and wait_for_user_input_callback:
            user_input = await wait_for_user_input_callback()
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

