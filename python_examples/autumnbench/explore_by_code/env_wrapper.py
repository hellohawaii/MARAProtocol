import atexit
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
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
INTERNAL_CONTROL_TOKEN = os.environ.get(
	"AUTUMNBENCH_INTERNAL_CONTROL_TOKEN",
	"autumnbench-internal-control-static-token",
)
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_SESSION: Optional["_DockerRuntime"] = None
_RUNTIME_KEY: Optional[tuple] = None
_BATCH_EVAL_ENV_CLIENT_TEMPLATE = """import os
import json
from typing import Any, Dict, Optional
from urllib import request, error


class RemoteEnvWrapper:
    def __init__(self, base_url: Optional[str] = None, timeout_seconds: int = 30):
        self.base_url = (base_url or os.getenv("ENV_API_BASE_URL") or "http://host.docker.internal:8002").rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_body = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} calling {url}: {message}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to call {url}: {exc}") from exc

        try:
            return json.loads(resp_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON response from {url}: {resp_body}") from exc

    def reset(self) -> Dict[str, Any]:
        raise RuntimeError("reset() is disabled in variant batch evaluator mode. The backend already prepared the environment.")

    def step(self, action: str):
        result = self._post("/step", {"action": action})
        if isinstance(result, dict) and "goal_reached" in result:
            return result["state"], result["goal_reached"]
        return result

    def save_trajectory(self, filename: Optional[str] = None) -> Dict[str, Any]:
        raise RuntimeError("save_trajectory() is disabled in variant batch evaluator mode. The backend saves trajectories after the agent finishes.")
"""


def _find_free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def _spawn_env_api_server(port: int) -> subprocess.Popen:
	server_script = str(BASE_DIR / "env_api_server.py")
	env = {**os.environ, "OBFUSCATED": "1"}
	proc = subprocess.Popen(
		[sys.executable, server_script, "--port", str(port)],
		env=env,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)
	deadline = time.time() + 10
	while time.time() < deadline:
		try:
			urlrequest.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
			return proc
		except Exception:
			time.sleep(0.2)
	proc.terminate()
	raise RuntimeError(f"env_api_server failed to start on port {port}")


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
		task_mode: str = "explore",
		template_dir: Optional[Path] = None,
		workspace_dir: Optional[Path] = None,
		spawn_env_server: bool = True,
	) -> None:
		self.docker_image = docker_image
		self.dockerfile_path = Path(dockerfile_path).resolve() if dockerfile_path else None
		self.docker_build_context = (
			Path(docker_build_context).resolve() if docker_build_context else None
		)
		self.env_name = env_name
		self.task_mode = task_mode
		self.template_dir = template_dir
		self.client = docker.from_env()
		self.container = None
		self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
		if workspace_dir is not None:
			self.active_workspace_dir = Path(workspace_dir).resolve()
		else:
			self.active_workspace_dir = (RUNS_WORKSPACE_ROOT_DIR / self.run_id).resolve()
		self.container_name = f"autumnbench-shell-{os.getpid()}-{uuid.uuid4().hex[:8]}"
		self.spawn_env_server = spawn_env_server
		if spawn_env_server:
			self._server_port = _find_free_port()
			self._server_process = _spawn_env_api_server(self._server_port)
			self.env_api_base_url = f"http://host.docker.internal:{self._server_port}"
		else:
			self._server_port = None
			self._server_process = None
			self.env_api_base_url = None
		atexit.register(self.close)

	def close(self) -> None:
		if self.container is not None:
			try:
				self.container.remove(force=True)
			except DockerException:
				pass
			self.container = None
		if getattr(self, "_server_process", None) is not None:
			try:
				self._server_process.terminate()
				self._server_process.wait(timeout=5)
			except Exception:
				pass
			self._server_process = None

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
		src = self.template_dir or TEMPLATE_WORKSPACE_DIR
		if not src.exists():
			raise ValueError(f"Template workspace not found: {src}")
		self.active_workspace_dir.parent.mkdir(parents=True, exist_ok=True)
		shutil.copytree(src, self.active_workspace_dir, dirs_exist_ok=False)
		(self.active_workspace_dir / "traj").mkdir(parents=True, exist_ok=True)

	def _post_env_api(
		self,
		path: str,
		payload: Optional[Dict[str, Any]] = None,
		*,
		internal: bool = False,
		expect_ok: bool = True,
	) -> Dict[str, Any]:
		url = f"http://127.0.0.1:{self._server_port}{path}"
		payload = payload or {}
		body = json.dumps(payload).encode("utf-8")
		headers = {"Content-Type": "application/json"}
		if internal:
			headers["X-Autumnbench-Internal-Token"] = INTERNAL_CONTROL_TOKEN
		req = urlrequest.Request(
			url,
			data=body,
			headers=headers,
			method="POST",
		)
		try:
			with urlrequest.urlopen(req, timeout=30) as resp:
				raw = resp.read().decode("utf-8")
		except urlerror.HTTPError as exc:
			msg = exc.read().decode("utf-8", errors="replace")
			raise ValueError(f"Failed to set env API workspace (HTTP {exc.code}): {msg}") from exc
		except urlerror.URLError as exc:
			raise ValueError(f"Failed to reach env API at {url}: {exc}") from exc

		try:
			parsed = json.loads(raw)
		except json.JSONDecodeError as exc:
			raise ValueError(f"Invalid {path} response: {raw}") from exc

		if expect_ok and not parsed.get("ok"):
			raise ValueError(f"Env API rejected {path}: {parsed}")
		return parsed

	def _get_env_api(self, path: str, *, internal: bool = False) -> Dict[str, Any]:
		url = f"http://127.0.0.1:{self._server_port}{path}"
		headers = {}
		if internal:
			headers["X-Autumnbench-Internal-Token"] = INTERNAL_CONTROL_TOKEN
		req = urlrequest.Request(url, headers=headers, method="GET")
		try:
			with urlrequest.urlopen(req, timeout=30) as resp:
				raw = resp.read().decode("utf-8")
		except urlerror.HTTPError as exc:
			msg = exc.read().decode("utf-8", errors="replace")
			raise ValueError(f"Failed calling {path} (HTTP {exc.code}): {msg}") from exc
		except urlerror.URLError as exc:
			raise ValueError(f"Failed to reach env API at {url}: {exc}") from exc

		try:
			return json.loads(raw)
		except json.JSONDecodeError as exc:
			raise ValueError(f"Invalid {path} response: {raw}") from exc

	def _set_env_api_workspace(self) -> None:
		self._post_env_api(
			"/set_workspace",
			{
				"workspace_dir": str(self.active_workspace_dir),
				"run_id": self.run_id,
			},
		)

	def _set_env_api_env_name(self) -> None:
		if not self.env_name:
			return
		self._post_env_api("/_set_env_name", {"env_name": self.env_name})

	def _set_env_api_task_mode(self) -> None:
		self._post_env_api("/_set_task_mode", {"task_mode": self.task_mode})

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
		if self.spawn_env_server:
			self._set_env_api_workspace()
			self._set_env_api_env_name()
			self._set_env_api_task_mode()

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

	def ensure_ready(self) -> None:
		self.ensure_daemon_ready()
		self._ensure_container_running()

	def configure_environment(
		self,
		*,
		env_name: Optional[str] = None,
		task_mode: Optional[str] = None,
	) -> None:
		if env_name is not None:
			self.env_name = env_name
		if task_mode is not None:
			self.task_mode = task_mode
		self.ensure_ready()
		if self.spawn_env_server:
			self._set_env_api_workspace()
			self._set_env_api_env_name()
			self._set_env_api_task_mode()

	def set_client_control_mode(self, mode: str) -> Dict[str, Any]:
		self.ensure_ready()
		return self._post_env_api(
			"/_set_client_control_mode",
			{"mode": mode},
			internal=True,
		)

	def backend_reset(self) -> Dict[str, Any]:
		self.ensure_ready()
		return self._post_env_api("/_backend_reset", {}, internal=True, expect_ok=False)

	def backend_save_trajectory(self, filename: Optional[str] = None) -> Dict[str, Any]:
		payload: Dict[str, Any] = {}
		if filename:
			payload["filename"] = filename
		self.ensure_ready()
		return self._post_env_api(
			"/_backend_save_trajectory",
			payload,
			internal=True,
			expect_ok=False,
		)

	def backend_get_trajectory(self) -> Dict[str, Any]:
		self.ensure_ready()
		return self._get_env_api("/_backend_get_trajectory", internal=True)

	def backend_goal_status(self) -> Dict[str, Any]:
		self.ensure_ready()
		return self._get_env_api("/_backend_goal_status", internal=True)

	def install_batch_eval_env_client(self) -> Path:
		self._prepare_run_workspace()
		out_path = self.active_workspace_dir / "env_api_client.py"
		out_path.write_text(_BATCH_EVAL_ENV_CLIENT_TEMPLATE, encoding="utf-8")
		return out_path

	def exec_command(self, command: str, timeout_seconds: int) -> dict:
		self.ensure_ready()

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


def _get_runtime(
	docker_image: Optional[str],
	dockerfile_path: Optional[str],
	docker_build_context: Optional[str],
	env_api_base_url: str,
	env_name: Optional[str],
	task_mode: str = "explore",
) -> _DockerRuntime:
	global _RUNTIME_SESSION
	global _RUNTIME_KEY
	requested_key = (
		docker_image,
		str(Path(dockerfile_path).resolve()) if dockerfile_path else None,
		str(Path(docker_build_context).resolve()) if docker_build_context else None,
		env_api_base_url,
		env_name,
		task_mode,
	)
	with _RUNTIME_LOCK:
		if _RUNTIME_SESSION is not None and _RUNTIME_KEY == requested_key:
			return _RUNTIME_SESSION

		candidate = _DockerRuntime(
			docker_image=docker_image,
			dockerfile_path=dockerfile_path,
			docker_build_context=docker_build_context,
			env_api_base_url=env_api_base_url,
			env_name=env_name,
			task_mode=task_mode,
		)
		if _RUNTIME_SESSION is None:
			_RUNTIME_SESSION = candidate
			_RUNTIME_KEY = requested_key
			return _RUNTIME_SESSION

		_RUNTIME_SESSION.close()
		_RUNTIME_SESSION = candidate
		_RUNTIME_KEY = requested_key
		return _RUNTIME_SESSION


def execute_run_command(
	command: str,
	timeout_seconds: int = 15,
	docker_image: Optional[str] = None,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8002",
	env_name: Optional[str] = None,
	task_mode: str = "explore",
) -> str:
	ensure_workspace_dirs()
	ts_start = time.time()
	ts_start_iso = datetime.fromtimestamp(ts_start).isoformat()
	try:
		runtime = _get_runtime(
			docker_image=docker_image,
			dockerfile_path=dockerfile_path,
			docker_build_context=docker_build_context,
			env_api_base_url=env_api_base_url,
			env_name=env_name,
			task_mode=task_mode,
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


def get_runtime_info() -> dict:
	with _RUNTIME_LOCK:
		if _RUNTIME_SESSION is None:
			return {}
		return _RUNTIME_SESSION.get_runtime_info()


def get_or_create_runtime_info(
	docker_image: Optional[str] = None,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8002",
	env_name: Optional[str] = None,
	task_mode: str = "explore",
) -> dict:
	ensure_workspace_dirs()
	runtime = _get_runtime(
		docker_image=docker_image,
		dockerfile_path=dockerfile_path,
		docker_build_context=docker_build_context,
		env_api_base_url=env_api_base_url,
		env_name=env_name,
		task_mode=task_mode,
	)
	return runtime.get_runtime_info()


def create_pinned_runtime(
	docker_image: Optional[str] = None,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8002",
	env_name: Optional[str] = None,
	task_mode: str = "explore",
	template_dir: Optional[Path] = None,
	workspace_dir: Optional[Path] = None,
	spawn_env_server: bool = True,
) -> _DockerRuntime:
	ensure_workspace_dirs()
	runtime = _DockerRuntime(
		docker_image=docker_image,
		dockerfile_path=dockerfile_path,
		docker_build_context=docker_build_context,
		env_api_base_url=env_api_base_url,
		env_name=env_name,
		task_mode=task_mode,
		template_dir=template_dir,
		workspace_dir=workspace_dir,
		spawn_env_server=spawn_env_server,
	)
	runtime.ensure_ready()
	return runtime


def get_env_tools_for_runtime(
	runtime: _DockerRuntime,
	timeout_seconds: int = 15,
):
	try:
		from langchain_core.tools import StructuredTool
	except ImportError as exc:
		raise ImportError(
			"langchain_core is required to build tools. Install langchain-core first."
		) from exc

	import asyncio

	def _run_command(command: str) -> str:
		ts_start = time.time()
		ts_start_iso = datetime.fromtimestamp(ts_start).isoformat()
		try:
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
				msg = f"✅ Command executed successfully.\n{output}" if output else "✅ Command executed successfully."
				if logging_errors:
					msg += "\n⚠️ Logging warnings:\n" + "\n".join(logging_errors)
				return msg

			msg = (
				f"❌ Command failed (exit code {result['exit_code']}).\n{output}"
				if output
				else f"❌ Command failed (exit code {result['exit_code']})."
			)
			if logging_errors:
				msg += "\n⚠️ Logging warnings:\n" + "\n".join(logging_errors)
			return msg
		except (DockerException, ValueError) as exc:
			return f"❌ Docker SDK error: {str(exc)}"
		except Exception as exc:
			return f"❌ Internal execution error: {str(exc)}\n{traceback.format_exc()}"

	async def _arun_command(command: str) -> str:
		return await asyncio.to_thread(_run_command, command)

	run_command_tool = StructuredTool.from_function(
		func=_run_command,
		coroutine=_arun_command,
		name="run_command_in_docker",
		description=(
			"Execute one shell command in a persistent Docker workspace at /workspace. "
			"Use this to run Python scripts, call RemoteEnvWrapper env.step(...), "
			"write code files, and inspect results. Container state and files persist across calls. "
			"Input: command."
		),
	)

	return [run_command_tool]


def get_env_tools(
	docker_image: Optional[str] = None,
	timeout_seconds: int = 15,
	dockerfile_path: Optional[str] = None,
	docker_build_context: Optional[str] = None,
	env_api_base_url: str = "http://host.docker.internal:8002",
	env_name: Optional[str] = None,
	task_mode: str = "explore",
):
	try:
		from langchain_core.tools import StructuredTool
	except ImportError as exc:
		raise ImportError(
			"langchain_core is required to build tools. Install langchain-core first."
		) from exc

	import asyncio

	def _run_command(command: str) -> str:
		return execute_run_command(
			command=command,
			timeout_seconds=timeout_seconds,
			docker_image=docker_image,
			dockerfile_path=dockerfile_path,
			docker_build_context=docker_build_context,
			env_api_base_url=env_api_base_url,
			env_name=env_name,
			task_mode=task_mode,
		)

	async def _arun_command(command: str) -> str:
		return await asyncio.to_thread(_run_command, command)

	run_command_tool = StructuredTool.from_function(
		func=_run_command,
		coroutine=_arun_command,
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
	"create_pinned_runtime",
	"get_runtime_info",
	"get_or_create_runtime_info",
	"get_env_tools_for_runtime",
	"get_env_tools",
]
