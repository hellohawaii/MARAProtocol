import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def _next_prefixed_index(parent_dir: Path) -> int:
    if not parent_dir.exists():
        return 1
    max_idx = 0
    for item in parent_dir.iterdir():
        if not item.is_dir():
            continue
        prefix, sep, _rest = item.name.partition("_")
        if not sep:
            continue
        try:
            num = int(prefix)
        except ValueError:
            continue
        if num > max_idx:
            max_idx = num
    return max_idx + 1


def create_subgraph_dir(parent_dir: str, name: str) -> str:
    """Create a subgraph directory like 001_perfect_zone_20260314_100000"""
    parent_path = Path(parent_dir)
    parent_path.mkdir(parents=True, exist_ok=True)
    seq = _next_prefixed_index(parent_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{seq:03d}_{name}_{timestamp}"
    new_dir = parent_path / dir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    return str(new_dir)


def create_node_dir(parent_dir: str, name: str) -> str:
    """Create a node directory like 001_run_react_refine and initialize subfolders"""
    parent_path = Path(parent_dir)
    parent_path.mkdir(parents=True, exist_ok=True)
    seq = _next_prefixed_index(parent_path)
    dir_name = f"{seq:03d}_{name}"
    new_dir = parent_path / dir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize subfolders
    (new_dir / "llm").mkdir(exist_ok=True)
    (new_dir / "code").mkdir(exist_ok=True)
    (new_dir / "shell").mkdir(exist_ok=True)
    (new_dir / "traj").mkdir(exist_ok=True)
    (new_dir / "other_logs").mkdir(exist_ok=True)
    
    return str(new_dir)


def get_next_sequence_dir(parent_dir: str, prefix: str) -> str:
    """Create a sequence directory like 001_code or 002_targeted_collection"""
    parent_path = Path(parent_dir)
    parent_path.mkdir(parents=True, exist_ok=True)
    seq = _next_prefixed_index(parent_path)
    dir_name = f"{seq:03d}_{prefix}"
    new_dir = parent_path / dir_name
    new_dir.mkdir(parents=True, exist_ok=True)
    return str(new_dir)


class NodeLoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that logs LLM inputs, outputs, and tool calls to the node's llm directory."""
    
    def __init__(self, node_log_dir: str):
        self.node_log_dir = Path(node_log_dir)
        self.llm_dir = self.node_log_dir / "llm"
        self.llm_dir.mkdir(parents=True, exist_ok=True)
        self.call_count = 0

    def _get_next_file_path(self, prefix: str) -> Path:
        self.call_count += 1
        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = f"{self.call_count:03d}_{prefix}_{timestamp}.json"
        return self.llm_dir / filename
        
    def _message_to_dict(self, message: BaseMessage) -> Dict[str, Any]:
        return {
            "type": message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
        }

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs: Any,
    ) -> Any:
        """Run when Chat Model starts running."""
        filepath = self._get_next_file_path("llm_start")
        
        formatted_messages = []
        for seq in messages:
            formatted_messages.append([self._message_to_dict(m) for m in seq])
            
        data = {
            "event": "on_chat_model_start",
            "messages": formatted_messages,
            "kwargs": {k: str(v) for k, v in kwargs.items() if k != "parent_run_id"}
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        filepath = self._get_next_file_path("llm_end")
        
        generations = []
        for gen_seq in response.generations:
            seq_data = []
            for gen in gen_seq:
                gen_dict = {
                    "text": gen.text,
                }
                if hasattr(gen, "message"):
                    gen_dict["message"] = self._message_to_dict(gen.message)
                seq_data.append(gen_dict)
            generations.append(seq_data)
            
        data = {
            "event": "on_llm_end",
            "generations": generations,
            "llm_output": response.llm_output,
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
        filepath = self._get_next_file_path("tool_start")
        data = {
            "event": "on_tool_start",
            "tool_name": serialized.get("name", "unknown_tool"),
            "input_str": input_str,
            "kwargs": {k: str(v) for k, v in kwargs.items() if k != "parent_run_id"}
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        filepath = self._get_next_file_path("tool_end")
        data = {
            "event": "on_tool_end",
            "output": output,
            "kwargs": {k: str(v) for k, v in kwargs.items() if k != "parent_run_id"}
        }
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
