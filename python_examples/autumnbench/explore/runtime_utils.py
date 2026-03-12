"""
Shared runtime helpers for explore graphs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from shared_runtime import convert_env_trajectory, parse_action_string, TrajectoryRecord


def message_content_to_text(content: Any) -> str:
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


def extract_stdout_text(command_result: str) -> str:
    marker = "[STDOUT]:"
    idx = command_result.find(marker)
    if idx < 0:
        return command_result
    text = command_result[idx + len(marker) :]
    stderr_idx = text.find("[STDERR]:")
    if stderr_idx >= 0:
        text = text[:stderr_idx]
    return text.strip()


def extract_first_json_array(text: str) -> List[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    try:
        maybe = json.loads(stripped)
        if isinstance(maybe, list):
            return [str(x) for x in maybe]
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", stripped)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        return []
    return []


def normalize_saved_trajectory(payload: Dict[str, Any]) -> Optional[TrajectoryRecord]:
    transitions = payload.get("trajectory")
    if not isinstance(transitions, list) or not transitions:
        return None

    first = transitions[0]
    first_state = first.get("state") if isinstance(first, dict) else None
    if not isinstance(first_state, dict):
        return None

    env_traj: List[Dict[str, Any]] = [{"action": None, "rawFrame": first_state}]
    for row in transitions:
        if not isinstance(row, dict):
            continue
        action_raw = row.get("action")
        if isinstance(action_raw, str):
            action = parse_action_string(action_raw) or action_raw
        else:
            action = action_raw
        next_state = row.get("new_state")
        if not isinstance(next_state, dict):
            continue
        env_traj.append({"action": action, "rawFrame": next_state})

    if len(env_traj) <= 1:
        return None
    return convert_env_trajectory(env_traj)


def load_workspace_trajectories(workspace_dir: str, traj_files: List[str]) -> List[TrajectoryRecord]:
    root = Path(workspace_dir).resolve()
    out: List[TrajectoryRecord] = []
    for rel in traj_files:
        rel_clean = rel.lstrip("/")
        candidate = (root / rel_clean).resolve()
        if not str(candidate).startswith(str(root)):
            continue
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        normalized = normalize_saved_trajectory(payload)
        if normalized is not None:
            normalized["traj_path"] = rel
            out.append(normalized)
    return out


def trajectory_to_saved_payload(traj: TrajectoryRecord) -> Optional[Dict[str, Any]]:
    frames = traj.get("frames", [])
    actions = traj.get("frameActions", [])
    if not isinstance(frames, list) or len(frames) < 2 or not isinstance(actions, list):
        return None

    transitions: List[Dict[str, Any]] = []
    for prev_frame, action, next_frame in zip(frames[:-1], actions, frames[1:]):
        transitions.append(
            {
                "state": prev_frame.get("rawFrame") if isinstance(prev_frame, dict) else None,
                "action": action,
                "new_state": next_frame.get("rawFrame") if isinstance(next_frame, dict) else None,
            }
        )
    if not transitions:
        return None
    return {"trajectory": transitions}


def read_code_from_runtime(
    execute_run_command_fn: Callable[..., str],
    *,
    env_name: str,
    runtime_key: str,
    code_path: str,
    fallback_code: str = "",
    timeout_seconds: int = 30,
) -> str:
    read_cmd = (
        "python - <<'PY'\n"
        "import json\n"
        f"path = {json.dumps(code_path)}\n"
        "try:\n"
        "    print(open(path, 'r', encoding='utf-8').read())\n"
        "except Exception:\n"
        "    print('')\n"
        "PY"
    )
    raw = execute_run_command_fn(
        read_cmd,
        env_name=env_name,
        timeout_seconds=timeout_seconds,
        runtime_key=runtime_key,
    )
    text = extract_stdout_text(raw).strip()
    return text or (fallback_code or "")


def write_code_to_runtime(
    execute_run_command_fn: Callable[..., str],
    *,
    code_text: str,
    code_path: str,
    env_name: str,
    runtime_key: str,
    timeout_seconds: int = 30,
) -> str:
    json_blob = json.dumps(code_text or "", ensure_ascii=True)
    cmd = (
        "python - <<'PY'\n"
        "import json, os\n"
        f"code = json.loads({json.dumps(json_blob)})\n"
        f"path = {json.dumps(code_path)}\n"
        "parent = os.path.dirname(path)\n"
        "if parent:\n"
        "    os.makedirs(parent, exist_ok=True)\n"
        "with open(path, 'w', encoding='utf-8') as f:\n"
        "    f.write(code)\n"
        "print('synced_code', path, 'chars', len(code))\n"
        "PY"
    )
    return execute_run_command_fn(
        cmd,
        env_name=env_name,
        timeout_seconds=timeout_seconds,
        runtime_key=runtime_key,
    )


def write_trajectories_to_runtime(
    execute_run_command_fn: Callable[..., str],
    *,
    env_name: str,
    runtime_key: str,
    trajectories: List[TrajectoryRecord],
    file_prefix: str,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for traj in trajectories:
        row = trajectory_to_saved_payload(traj)
        if row is not None:
            rows.append(row)
    if not rows:
        return {"written_files": [], "runtime_sync_output": "no valid trajectories to write"}

    payload = json.dumps(rows, ensure_ascii=True)
    cmd = (
        "python - <<'PY'\n"
        "import json, os, glob\n"
        "os.makedirs('traj', exist_ok=True)\n"
        f"rows = json.loads({json.dumps(payload)})\n"
        f"prefix = {json.dumps(file_prefix)}\n"
        "existing_files = glob.glob(f'traj/{prefix}_*.json')\n"
        "max_idx = -1\n"
        "for f in existing_files:\n"
        "    try:\n"
        "        idx = int(os.path.basename(f).replace(f'{prefix}_', '').replace('.json', ''))\n"
        "        max_idx = max(max_idx, idx)\n"
        "    except ValueError:\n"
        "        pass\n"
        "start_idx = max_idx + 1\n"
        "written = []\n"
        "for i, row in enumerate(rows):\n"
        "    path = f'traj/{prefix}_{start_idx + i:03d}.json'\n"
        "    with open(path, 'w', encoding='utf-8') as f:\n"
        "        json.dump(row, f, ensure_ascii=True, indent=2)\n"
        "    written.append(path)\n"
        "print(json.dumps(written, ensure_ascii=True))\n"
        "PY"
    )
    raw = execute_run_command_fn(
        cmd,
        env_name=env_name,
        timeout_seconds=timeout_seconds,
        runtime_key=runtime_key,
    )
    written_files = extract_first_json_array(extract_stdout_text(raw).strip())
    return {"written_files": written_files, "runtime_sync_output": raw}


def load_all_trajectories_from_runtime(
    execute_run_command_fn: Callable[..., str],
    get_or_create_runtime_info_fn: Callable[..., Dict[str, Any]],
    *,
    env_name: str,
    runtime_key: str,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    list_cmd = (
        "python - <<'PY'\n"
        "import glob, json\n"
        "files = sorted(glob.glob('traj/**/*.json', recursive=True))\n"
        "print(json.dumps(files, ensure_ascii=True))\n"
        "PY"
    )
    list_raw = execute_run_command_fn(
        list_cmd,
        env_name=env_name,
        timeout_seconds=max(10, int(timeout_seconds)),
        runtime_key=runtime_key,
    )
    traj_files = extract_first_json_array(extract_stdout_text(list_raw))
    runtime_info = get_or_create_runtime_info_fn(env_name=env_name, runtime_key=runtime_key)
    workspace_dir = str(runtime_info.get("workspace_dir") or "").strip()
    if not workspace_dir:
        return {
            "all_trajectories": [],
            "all_traj_files": traj_files,
            "load_output": "runtime workspace missing",
        }
    all_trajectories = load_workspace_trajectories(workspace_dir, traj_files)
    return {
        "all_trajectories": all_trajectories,
        "all_traj_files": traj_files,
        "load_output": list_raw,
    }
