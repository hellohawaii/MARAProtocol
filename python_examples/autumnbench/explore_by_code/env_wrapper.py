import atexit
import hashlib
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

import docker
from docker.errors import DockerException, NotFound
try:
	from .log_utils import (
		append_jsonl,
		ensure_logs_root_dir,
		persist_check_artifacts,
		run_logs_dir,
	)
except ImportError:  # pragma: no cover
	from log_utils import (
		append_jsonl,
		ensure_logs_root_dir,
		persist_check_artifacts,
		run_logs_dir,
	)


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_WORKSPACE_DIR = (BASE_DIR / "llm_workspace").resolve()
RUNS_WORKSPACE_ROOT_DIR = (BASE_DIR / "llm_workspace_runs").resolve()
WORKSPACE_DIR = TEMPLATE_WORKSPACE_DIR
TRAJ_DIR = TEMPLATE_WORKSPACE_DIR / "traj"
DEFAULT_DOCKER_IMAGE = "python:3.11-slim"
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_SESSIONS: Dict[
	Tuple[Optional[str], Optional[str]],
	"_DockerRuntime",
] = {}


def ensure_workspace_dirs() -> None:
	TEMPLATE_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
	TRAJ_DIR.mkdir(parents=True, exist_ok=True)
	RUNS_WORKSPACE_ROOT_DIR.mkdir(parents=True, exist_ok=True)
	ensure_logs_root_dir()


class _DockerRuntime:
	def __init__(
		self,
		docker_image: Optional[str],
		dockerfile_path: Optional[str],
		docker_build_context: Optional[str],
		env_api_base_url: str,
		env_name: Optional[str],
	) -> None:
		self.docker_image = docker_image
		self.dockerfile_path = Path(dockerfile_path).resolve() if dockerfile_path else None
		self.docker_build_context = (
			Path(docker_build_context).resolve() if docker_build_context else None
		)
		self.env_api_base_url = env_api_base_url
		self.env_name = env_name
		self.client = docker.from_env()
		self.container = None
		self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		self.active_workspace_dir = (RUNS_WORKSPACE_ROOT_DIR / self.run_id).resolve()
		self.container_name = f"autumnbench-shell-{os.getpid()}-{uuid.uuid4().hex[:8]}"
		atexit.register(self.close)

	def close(self) -> None:
		if self.container is not None:
			try:
				self.container.remove(force=True)
			except DockerException:
				pass
			self.container = None

	def ensure_daemon_ready(self) -> None:
		self.client.ping()

	def _resolve_build_context_and_dockerfile(self) -> tuple:
		if self.dockerfile_path is None:
			raise ValueError("dockerfile_path is required when building custom image.")

		if not self.dockerfile_path.exists():
			raise ValueError(f"Dockerfile not found: {self.dockerfile_path}")

		build_context = self.docker_build_context or self.dockerfile_path.parent
		if not build_context.exists() or not build_context.is_dir():
			raise ValueError(f"Invalid docker build context: {build_context}")

		try:
			rel_dockerfile = self.dockerfile_path.relative_to(build_context)
		except ValueError as exc:
			raise ValueError(
				"Dockerfile must be located inside docker_build_context for SDK build."
			) from exc

		return build_context, str(rel_dockerfile)

	def _build_image_tag(self) -> str:
		payload = (
			f"{self.dockerfile_path}|{self.docker_build_context}|{TEMPLATE_WORKSPACE_DIR}"
		).encode("utf-8")
		return f"autumnbench/explore-shell:{hashlib.sha256(payload).hexdigest()[:12]}"

	def _ensure_image(self) -> str:
		if self.dockerfile_path is None:
			return self.docker_image or DEFAULT_DOCKER_IMAGE

		build_context, dockerfile_rel = self._resolve_build_context_and_dockerfile()
		image_tag = self._build_image_tag()
		try:
			self.client.images.get(image_tag)
			return image_tag
		except NotFound:
			pass

		self.client.images.build(
			path=str(build_context),
			dockerfile=dockerfile_rel,
			tag=image_tag,
			rm=True,
		)
		return image_tag

	def _prepare_run_workspace(self) -> None:
		if self.active_workspace_dir.exists():
			return
		if not TEMPLATE_WORKSPACE_DIR.exists():
			raise ValueError(f"Template workspace not found: {TEMPLATE_WORKSPACE_DIR}")
		RUNS_WORKSPACE_ROOT_DIR.mkdir(parents=True, exist_ok=True)
		shutil.copytree(TEMPLATE_WORKSPACE_DIR, self.active_workspace_dir, dirs_exist_ok=False)
		(self.active_workspace_dir / "traj").mkdir(parents=True, exist_ok=True)

	def _set_env_api_workspace(self) -> None:
		url = "http://127.0.0.1:8000/set_workspace"
		payload = {
			"workspace_dir": str(self.active_workspace_dir),
			"run_id": self.run_id,
		}
		body = json.dumps(payload).encode("utf-8")
		req = urlrequest.Request(
			url,
			data=body,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		try:
			with urlrequest.urlopen(req, timeout=10) as resp:
				raw = resp.read().decode("utf-8")
		except urlerror.HTTPError as exc:
			msg = exc.read().decode("utf-8", errors="replace")
			raise ValueError(f"Failed to set env API workspace (HTTP {exc.code}): {msg}") from exc
		except urlerror.URLError as exc:
			raise ValueError(f"Failed to reach env API at {url}: {exc}") from exc

		try:
			parsed = json.loads(raw)
		except json.JSONDecodeError as exc:
			raise ValueError(f"Invalid /set_workspace response: {raw}") from exc

		if not parsed.get("ok"):
			raise ValueError(f"Env API rejected workspace switch: {parsed}")

	def _set_env_api_env_name(self) -> None:
		if not self.env_name:
			return
		url = "http://127.0.0.1:8000/_set_env_name"
		payload = {"env_name": self.env_name}
		body = json.dumps(payload).encode("utf-8")
		req = urlrequest.Request(
			url,
			data=body,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		try:
			with urlrequest.urlopen(req, timeout=10) as resp:
				raw = resp.read().decode("utf-8")
		except urlerror.HTTPError as exc:
			msg = exc.read().decode("utf-8", errors="replace")
			raise ValueError(f"Failed to set hidden env name (HTTP {exc.code}): {msg}") from exc
		except urlerror.URLError as exc:
			raise ValueError(f"Failed to reach env API at {url}: {exc}") from exc

		try:
			parsed = json.loads(raw)
		except json.JSONDecodeError as exc:
			raise ValueError(f"Invalid /_set_env_name response: {raw}") from exc

		if not parsed.get("ok"):
			raise ValueError(f"Env API rejected env_name switch: {parsed}")

	def _ensure_container_running(self) -> None:
		if self.container is not None:
			try:
				self.container.reload()
				if self.container.status == "running":
					return
				self.container.start()
				return
			except DockerException:
				self.container = None

		image = self._ensure_image()
		if hasattr(os, "getuid") and hasattr(os, "getgid"):
			user_id = f"{os.getuid()}:{os.getgid()}"
		else:
			user_id = "1000:1000"
		self._prepare_run_workspace()
		self._set_env_api_workspace()
		self._set_env_api_env_name()

		try:
			self.container = self.client.containers.run(
				image=image,
				command=["tail", "-f", "/dev/null"],
				name=self.container_name,
				detach=True,
				volumes={str(self.active_workspace_dir): {"bind": "/workspace", "mode": "rw"}},
				working_dir="/workspace",
				environment={
					"ENV_API_BASE_URL": self.env_api_base_url,
					"RUN_ID": self.run_id,
					"RUN_WORKSPACE_DIR": str(self.active_workspace_dir),
				},
				extra_hosts={"host.docker.internal": "host-gateway"},
				mem_limit="512m",
				memswap_limit="512m",
				nano_cpus=1_000_000_000,
				pids_limit=50,
				security_opt=["no-new-privileges:true"],
				user=user_id,
			)
		except DockerException:
			# If name collision happens after abnormal process exit, reuse that container.
			self.container = self.client.containers.get(self.container_name)
			self.container.start()

	def exec_command(self, command: str, timeout_seconds: int) -> dict:
		self.ensure_daemon_ready()
		self._ensure_container_running()

		state = {}
		done = threading.Event()

		def _run_exec() -> None:
			try:
				exit_code, output = self.container.exec_run(
					cmd=["sh", "-lc", command],
					demux=True,
				)
				stdout_bytes, stderr_bytes = output if output else (b"", b"")
				state["exit_code"] = exit_code
				state["stdout"] = (stdout_bytes or b"").decode("utf-8", errors="replace")
				state["stderr"] = (stderr_bytes or b"").decode("utf-8", errors="replace")
			except Exception as exc:
				state["error"] = exc
			finally:
				done.set()

		thread = threading.Thread(target=_run_exec, daemon=True)
		thread.start()
		if not done.wait(timeout_seconds):
			return {
				"timed_out": True,
				"stdout": "",
				"stderr": "",
				"exit_code": None,
			}

		if "error" in state:
			raise state["error"]

		return {
			"timed_out": False,
			"stdout": state.get("stdout", ""),
			"stderr": state.get("stderr", ""),
			"exit_code": state.get("exit_code"),
		}

	def get_runtime_info(self) -> dict:
		return {
			"run_id": self.run_id,
			"workspace_dir": str(self.active_workspace_dir),
			"container_name": self.container_name,
			"container_running": bool(self.container is not None),
		}


def _make_runtime_request_key(
	docker_image: Optional[str],
	dockerfile_path: Optional[str],
	docker_build_context: Optional[str],
	env_api_base_url: str,
	env_name: Optional[str],
	runtime_key: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
	return (
		env_name,
		runtime_key,
	)


def _get_runtime(
	docker_image: Optional[str],
	dockerfile_path: Optional[str],
	docker_build_context: Optional[str],
	env_api_base_url: str,
	env_name: Optional[str],
	runtime_key: Optional[str] = None,
) -> _DockerRuntime:
	requested_key = _make_runtime_request_key(
		docker_image=docker_image,
		dockerfile_path=dockerfile_path,
		docker_build_context=docker_build_context,
		env_api_base_url=env_api_base_url,
		env_name=env_name,
		runtime_key=runtime_key,
	)
	with _RUNTIME_LOCK:
		existing = _RUNTIME_SESSIONS.get(requested_key)
		if existing is not None:
			return existing
		candidate = _DockerRuntime(
			docker_image=docker_image,
			dockerfile_path=dockerfile_path,
			docker_build_context=docker_build_context,
			env_api_base_url=env_api_base_url,
			env_name=env_name,
		)
		_RUNTIME_SESSIONS[requested_key] = candidate
		return candidate


def close_runtime(
	docker_image: Optional[str] = None,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8000",
	env_name: Optional[str] = None,
	runtime_key: Optional[str] = None,
) -> None:
	requested_key = _make_runtime_request_key(
		docker_image=docker_image,
		dockerfile_path=dockerfile_path,
		docker_build_context=docker_build_context,
		env_api_base_url=env_api_base_url,
		env_name=env_name,
		runtime_key=runtime_key,
	)
	with _RUNTIME_LOCK:
		runtime = _RUNTIME_SESSIONS.pop(requested_key, None)
	if runtime is not None:
		runtime.close()


def execute_run_command(
	command: str,
	timeout_seconds: int = 15,
	docker_image: Optional[str] = None,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8000",
	env_name: Optional[str] = None,
	runtime_key: Optional[str] = None,
	log_dir: Optional[str] = None,
) -> str:
	ensure_workspace_dirs()
	
	actual_log_dir = log_dir
	if log_dir:
		log_path = Path(log_dir)
		if log_path.exists() and log_path.name.endswith("_run_react_refine"):
			max_idx = -1
			latest_dir = None
			for item in log_path.iterdir():
				if not item.is_dir():
					continue
				prefix, sep, _rest = item.name.partition("_")
				if not sep:
					continue
				try:
					num = int(prefix)
					if num > max_idx:
						max_idx = num
						latest_dir = item
				except ValueError:
					continue
			if latest_dir and latest_dir.name.endswith("_code"):
				actual_log_dir = str(latest_dir)
			else:
				seq = max_idx + 1 if max_idx >= 0 else 1
				new_dir = log_path / f"{seq:03d}_code"
				new_dir.mkdir(parents=True, exist_ok=True)
				actual_log_dir = str(new_dir)
	ts_start = time.time()
	ts_start_iso = datetime.fromtimestamp(ts_start).isoformat()
	try:
		runtime = _get_runtime(
			docker_image=docker_image,
			dockerfile_path=dockerfile_path,
			docker_build_context=docker_build_context,
			env_api_base_url=env_api_base_url,
			env_name=env_name,
			runtime_key=runtime_key,
		)
		result = runtime.exec_command(command=command, timeout_seconds=timeout_seconds)
		ts_end = time.time()
		ts_end_iso = datetime.fromtimestamp(ts_end).isoformat()

		logging_errors: List[str] = []
		record = {
			"run_id": runtime.run_id,
			"workspace_dir": str(runtime.active_workspace_dir),
			"ts_start": ts_start_iso,
			"ts_end": ts_end_iso,
			"duration_ms": int((ts_end - ts_start) * 1000),
			"command": command,
			"timeout_seconds": timeout_seconds,
			"timed_out": bool(result.get("timed_out")),
			"exit_code": result.get("exit_code"),
			"stdout": result.get("stdout", ""),
			"stderr": result.get("stderr", ""),
		}
		try:
			if actual_log_dir:
				append_jsonl(Path(actual_log_dir) / "shell" / "commands.jsonl", record)
			else:
				append_jsonl(run_logs_dir(runtime.run_id) / "commands.jsonl", record)
		except Exception as exc:
			logging_errors.append(f"failed to append commands.jsonl: {exc}")

		try:
			persist_check_artifacts(
				command=command,
				result=result,
				run_id=runtime.run_id,
				workspace_dir=runtime.active_workspace_dir,
				ts_start_iso=ts_start_iso,
				ts_end_iso=ts_end_iso,
				log_dir=actual_log_dir,
			)
		except Exception as exc:
			logging_errors.append(f"failed to persist check artifacts: {exc}")

		if result["timed_out"]:
			msg = (
				f"⏰ Timeout: command exceeded {timeout_seconds} seconds. "
				"Container remains alive and command may still be running."
			)
			if logging_errors:
				msg += "\n⚠️ Logging warnings:\n" + "\n".join(logging_errors)
			return msg

		output_parts = []
		if result["stdout"]:
			output_parts.append(f"[STDOUT]:\n{result['stdout']}")
		if result["stderr"]:
			output_parts.append(f"[STDERR]:\n{result['stderr']}")
		output = "\n".join(output_parts).strip()

		if result["exit_code"] == 0:
			if output:
				msg = f"✅ Command executed successfully.\n{output}"
			else:
				msg = "✅ Command executed successfully."
			if logging_errors:
				msg += "\n⚠️ Logging warnings:\n" + "\n".join(logging_errors)
			return msg

		if output:
			msg = f"❌ Command failed (exit code {result['exit_code']}).\n{output}"
		else:
			msg = f"❌ Command failed (exit code {result['exit_code']})."
		if logging_errors:
			msg += "\n⚠️ Logging warnings:\n" + "\n".join(logging_errors)
		return msg
	except (DockerException, ValueError) as exc:
		return f"❌ Docker SDK error: {str(exc)}"
	except Exception as exc:
		return f"❌ Internal execution error: {str(exc)}\n{traceback.format_exc()}"


def get_runtime_info(
	docker_image: Optional[str] = None,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8000",
	env_name: Optional[str] = None,
	runtime_key: Optional[str] = None,
) -> dict:
	requested_key = _make_runtime_request_key(
		docker_image=docker_image,
		dockerfile_path=dockerfile_path,
		docker_build_context=docker_build_context,
		env_api_base_url=env_api_base_url,
		env_name=env_name,
		runtime_key=runtime_key,
	)
	with _RUNTIME_LOCK:
		runtime = _RUNTIME_SESSIONS.get(requested_key)
	return runtime.get_runtime_info() if runtime is not None else {}


def get_or_create_runtime_info(
	docker_image: Optional[str] = None,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8000",
	env_name: Optional[str] = None,
	runtime_key: Optional[str] = None,
) -> dict:
	ensure_workspace_dirs()
	runtime = _get_runtime(
		docker_image=docker_image,
		dockerfile_path=dockerfile_path,
		docker_build_context=docker_build_context,
		env_api_base_url=env_api_base_url,
		env_name=env_name,
		runtime_key=runtime_key,
	)
	return runtime.get_runtime_info()


def get_langchain_tools(
	docker_image: Optional[str] = None,
	timeout_seconds: int = 15,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8000",
	env_name: Optional[str] = None,
	runtime_key: Optional[str] = None,
	log_dir: Optional[str] = None,
):
	try:
		from langchain_core.tools import StructuredTool
	except ImportError as exc:
		raise ImportError(
			"langchain_core is required to build tools. Install langchain-core first."
		) from exc

	def _run_command(command: str) -> str:
		return execute_run_command(
			command=command,
			timeout_seconds=timeout_seconds,
			docker_image=docker_image,
			dockerfile_path=dockerfile_path,
			docker_build_context=docker_build_context,
			env_api_base_url=env_api_base_url,
			env_name=env_name,
			runtime_key=runtime_key,
			log_dir=log_dir,
		)

	run_command_tool = StructuredTool.from_function(
		func=_run_command,
		name="run_command_in_docker",
		description=(
			"Execute one shell command in a persistent Docker workspace at /workspace. "
			"Use this to run Python scripts, call RemoteEnvWrapper (env.reset/env.step/env.save_trajectory), "
			"write code files, and run check_traj_example.py on traj/ data. "
			"Container state and files persist across calls. Input: command."
		),
	)

	return [run_command_tool]


__all__: List[str] = [
	"WORKSPACE_DIR",
	"TRAJ_DIR",
	"DEFAULT_DOCKER_IMAGE",
	"ensure_workspace_dirs",
	"execute_run_command",
	"close_runtime",
	"get_runtime_info",
	"get_or_create_runtime_info",
	"get_langchain_tools",
]
