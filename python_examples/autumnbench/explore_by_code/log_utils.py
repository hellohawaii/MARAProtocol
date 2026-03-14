import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


BASE_DIR = Path(__file__).resolve().parent
LOGS_ROOT_DIR = (BASE_DIR / "logs").resolve()
_CHECK_TRAJ_PATTERN = re.compile(
    r"(?:^|[\s;&|])(?:(?:python|python3)\s+)?(?:\./)?check_traj_example\.py\s+"
    r"(?P<code>(?:\"[^\"]+\"|'[^']+'|[^\s;&|]+))\s+"
    r"(?P<traj>(?:\"[^\"]+\"|'[^']+'|[^\s;&|]+))"
)


def ensure_logs_root_dir() -> None:
    LOGS_ROOT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def run_logs_dir(run_id: Optional[str]) -> Path:
    out_dir = LOGS_ROOT_DIR / safe_name(run_id or "no_run_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def timeline_dir(run_id: Optional[str], log_dir: Optional[str] = None) -> Path:
    if log_dir:
        out_dir = Path(log_dir) / "code" / "timeline"
    else:
        out_dir = run_logs_dir(run_id) / "timeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _next_prefixed_index(parent_dir: Path) -> int:
    if not parent_dir.exists():
        return 1
    max_idx = 0
    for item in parent_dir.iterdir():
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


def allocate_timeline_index(run_id: Optional[str], log_dir: Optional[str] = None) -> int:
    tdir = timeline_dir(run_id, log_dir=log_dir)
    lock_path = tdir / ".index.lock"
    counter_path = tdir / ".index.counter"

    def _compute_next() -> int:
        if counter_path.exists():
            raw = counter_path.read_text(encoding="utf-8").strip()
            try:
                return int(raw) + 1
            except ValueError:
                pass
        return _next_prefixed_index(tdir)

    if fcntl is not None:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            next_idx = _compute_next()
            counter_path.write_text(str(next_idx), encoding="utf-8")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return next_idx

    # Fallback without file lock.
    next_idx = _compute_next()
    counter_path.write_text(str(next_idx), encoding="utf-8")
    return next_idx


def timeline_copy_file(
    *,
    run_id: Optional[str],
    index: int,
    stem: str,
    source_path: Path,
    suffix: str,
    log_dir: Optional[str] = None,
) -> Path:
    dest = timeline_dir(run_id, log_dir=log_dir) / f"{index}_{safe_name(stem)}{suffix}"
    shutil.copy2(source_path, dest)
    return dest


def timeline_write_json(
    *,
    run_id: Optional[str],
    index: int,
    stem: str,
    payload: Dict[str, Any],
    log_dir: Optional[str] = None,
) -> Path:
    dest = timeline_dir(run_id, log_dir=log_dir) / f"{index}_{safe_name(stem)}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def timeline_write_python_snapshot(
    *,
    run_id: Optional[str],
    index: int,
    stem: str,
    source_path: Optional[Path],
    log_dir: Optional[str] = None,
) -> Path:
    dest = timeline_dir(run_id, log_dir=log_dir) / f"{index}_{safe_name(stem)}.py"
    if source_path is not None and source_path.is_file():
        shutil.copy2(source_path, dest)
        return dest
    dest.write_text("# Snapshot unavailable: source file not found.\n", encoding="utf-8")
    return dest


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False))
        f.write("\n")


def _strip_shell_quotes(value: str) -> str:
    if len(value) >= 2 and ((value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"')):
        return value[1:-1]
    return value


def resolve_workspace_path(workspace_dir: Path, path_text: str) -> Path:
    clean = _strip_shell_quotes(path_text)
    if clean.startswith("/workspace/"):
        rel = clean[len("/workspace/"):]
        return (workspace_dir / rel).resolve()
    if clean == "/workspace":
        return workspace_dir.resolve()
    return (workspace_dir / clean).resolve()


def extract_json_from_stdout(stdout: str) -> Optional[Any]:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for idx in range(len(text)):
        if text[idx] != "{":
            continue
        try:
            return json.loads(text[idx:])
        except json.JSONDecodeError:
            continue
    return None


def persist_check_artifacts(
    *,
    command: str,
    result: Dict[str, Any],
    run_id: str,
    workspace_dir: Path,
    ts_start_iso: str,
    ts_end_iso: str,
    log_dir: Optional[str] = None,
) -> None:
    match = _CHECK_TRAJ_PATTERN.search(command)
    if not match:
        return

    code_path_token = match.group("code")
    traj_path_token = match.group("traj")
    code_path = resolve_workspace_path(workspace_dir, code_path_token)
    traj_path = resolve_workspace_path(workspace_dir, traj_path_token)

    if log_dir:
        checks_root = Path(log_dir) / "code" / "checks"
    else:
        checks_root = run_logs_dir(run_id) / "checks"
    checks_root.mkdir(parents=True, exist_ok=True)
    seq = _next_prefixed_index(checks_root)
    event_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    check_dir = checks_root / f"{seq}_{event_id}"
    check_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "workspace_dir": str(workspace_dir),
        "event_id": event_id,
        "command": command,
        "ts_start": ts_start_iso,
        "ts_end": ts_end_iso,
        "code_path_token": code_path_token,
        "traj_path_token": traj_path_token,
        "resolved_code_path": str(code_path),
        "resolved_traj_path": str(traj_path),
    }
    (check_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if code_path.is_file():
        prefixed_code_name = f"{seq}_{code_path.name}"
        shutil.copy2(code_path, check_dir / prefixed_code_name)

    result_payload: Dict[str, Any] = {
        "timed_out": bool(result.get("timed_out")),
        "exit_code": result.get("exit_code"),
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }
    parsed = extract_json_from_stdout(result_payload["stdout"])
    if parsed is not None:
        result_payload["parsed_stdout_json"] = parsed

    (check_dir / "result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Unified timeline artifacts with the same sequence id.
    timeline_stem = safe_name(code_path.stem if code_path.suffix else code_path.name)
    timeline_idx = allocate_timeline_index(run_id, log_dir=log_dir)
    timeline_write_python_snapshot(
        run_id=run_id,
        index=timeline_idx,
        stem=timeline_stem,
        source_path=code_path if code_path.is_file() else None,
        log_dir=log_dir,
    )
    timeline_write_json(
        run_id=run_id,
        index=timeline_idx,
        stem=timeline_stem,
        payload=result_payload,
        log_dir=log_dir,
    )
