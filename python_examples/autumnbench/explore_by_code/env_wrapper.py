import os
import subprocess
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = (BASE_DIR / "llm_workspace").resolve()
TRAJ_DIR = WORKSPACE_DIR / "traj"
DEFAULT_DOCKER_IMAGE = "python:3.11-slim"


def ensure_workspace_dirs() -> None:
	WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
	TRAJ_DIR.mkdir(parents=True, exist_ok=True)


def execute_write_file(filename: str, content: str) -> str:
	ensure_workspace_dirs()
	safe_filename = os.path.basename(filename)
	file_path = WORKSPACE_DIR / safe_filename
	file_path.write_text(content, encoding="utf-8")
	return f"✅ File {safe_filename} was saved to workspace successfully."


def execute_run_command(
	command: str,
	timeout_seconds: int = 15,
	docker_image: str = DEFAULT_DOCKER_IMAGE,
	env_api_base_url: str = "http://host.docker.internal:8000",
) -> str:
	ensure_workspace_dirs()

	if hasattr(os, "getuid") and hasattr(os, "getgid"):
		user_id = f"{os.getuid()}:{os.getgid()}"
	else:
		user_id = "1000:1000"

	docker_cmd = [
		"docker",
		"run",
		"--rm",
		"-v",
		f"{str(WORKSPACE_DIR)}:/workspace",
		"-w",
		"/workspace",
		"--add-host",
		"host.docker.internal:host-gateway",
		"--memory",
		"512m",
		"--memory-swap",
		"512m",
		"--cpus",
		"1.0",
		"--pids-limit",
		"50",
		"--security-opt",
		"no-new-privileges",
		"--user",
		user_id,
		"-e",
		f"ENV_API_BASE_URL={env_api_base_url}",
		docker_image,
		"sh",
		"-lc",
		command,
	]

	try:
		result = subprocess.run(
			docker_cmd,
			capture_output=True,
			text=True,
			timeout=timeout_seconds,
		)

		output_parts = []
		if result.stdout:
			output_parts.append(f"[STDOUT]:\n{result.stdout}")
		if result.stderr:
			output_parts.append(f"[STDERR]:\n{result.stderr}")
		output = "\n".join(output_parts).strip()

		if result.returncode == 0:
			if output:
				return f"✅ Command executed successfully.\n{output}"
			return "✅ Command executed successfully."

		if output:
			return f"❌ Command failed (exit code {result.returncode}).\n{output}"
		return f"❌ Command failed (exit code {result.returncode})."
	except subprocess.TimeoutExpired:
		return f"⏰ Timeout: command exceeded {timeout_seconds} seconds and was terminated. Please check for infinite loops."
	except Exception as exc:
		return f"❌ Internal execution error: {str(exc)}"


def get_langchain_tools(
	docker_image: str = DEFAULT_DOCKER_IMAGE,
	timeout_seconds: int = 15,
):
	try:
		from langchain_core.tools import StructuredTool
	except ImportError as exc:
		raise ImportError(
			"langchain_core is required to build tools. Install langchain-core first."
		) from exc

	def _write_file(filename: str, content: str) -> str:
		return execute_write_file(filename=filename, content=content)

	def _run_command(command: str) -> str:
		return execute_run_command(
			command=command,
			timeout_seconds=timeout_seconds,
			docker_image=docker_image,
		)

	write_file_tool = StructuredTool.from_function(
		func=_write_file,
		name="write_file",
		description=(
			"Write content into a file in llm_workspace with filename sandboxed by basename. "
			"Input args: filename, content."
		),
	)

	run_command_tool = StructuredTool.from_function(
		func=_run_command,
		name="run_command_in_docker",
		description=(
			"Execute a shell command inside isolated Docker, with /workspace mounted to llm_workspace. "
			"Input arg: command."
		),
	)

	return [write_file_tool, run_command_tool]


__all__: List[str] = [
	"WORKSPACE_DIR",
	"TRAJ_DIR",
	"DEFAULT_DOCKER_IMAGE",
	"ensure_workspace_dirs",
	"execute_write_file",
	"execute_run_command",
	"get_langchain_tools",
]
