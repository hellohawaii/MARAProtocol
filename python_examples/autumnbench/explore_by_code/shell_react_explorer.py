"""Minimal shell-tool ReAct explorer + code synthesis agent.

This script intentionally does not encode a workflow for exploration/refinement.
The LLM gets one shell tool and decides how to explore, write Python files,
run check_traj_example.py, and when to stop.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

_FILE_DIR = Path(__file__).resolve().parent
_AUTUMNBENCH_DIR = _FILE_DIR.parent
_MARA_ROOT = _AUTUMNBENCH_DIR.parents[1]
for _p in [str(_FILE_DIR), str(_AUTUMNBENCH_DIR), str(_MARA_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_utils import get_llm  # noqa: E402

from env_wrapper import (  # noqa: E402
    get_langchain_tools,
    get_or_create_runtime_info,
)
from shell_react_prompt import (  # noqa: E402
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


def create_shell_react_agent(
    env_name: str,
    llm_model: str = "google/gemini-2.5-pro",
    timeout_seconds: int = 30,
    docker_image: Optional[str] = None,
    dockerfile_path: Optional[str] = DEFAULT_DOCKERFILE_PATH,
    docker_build_context: Optional[str] = None,
    env_api_base_url: str = "http://host.docker.internal:8000",
    checkpointer: Optional[Any] = None,
):
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
        system_prompt=SHELL_REACT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent, runtime_info


def run_shell_react_agent(
    env_name: str,
    *,
    llm_model: str = "google/gemini-2.5-pro",
    max_turns: int = 120,
    timeout_seconds: int = 30,
    docker_image: Optional[str] = None,
    dockerfile_path: Optional[str] = DEFAULT_DOCKERFILE_PATH,
    docker_build_context: Optional[str] = None,
    env_api_base_url: str = "http://host.docker.internal:8000",
) -> Dict[str, Any]:
    agent, runtime_info = create_shell_react_agent(
        env_name=env_name,
        llm_model=llm_model,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        dockerfile_path=dockerfile_path,
        docker_build_context=docker_build_context,
        env_api_base_url=env_api_base_url,
    )

    user_prompt = build_initial_user_prompt(env_name)

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

    final_messages: List[Any] = []
    stream_event_count = 0
    for event in agent.stream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"recursion_limit": max_turns, "callbacks": [trace_cb]},
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
        default=120,
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
        default="http://host.docker.internal:8000",
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

