import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool
from langchain_core.tools import ToolException

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MARA_PROTOCOL_ROOT = _REPO_ROOT / "MARAProtocol"
_EXTRA_PATHS = [
	_REPO_ROOT,
	_MARA_PROTOCOL_ROOT,
]
for path in _EXTRA_PATHS:
	if path.exists():
		path_str = str(path)
		if path_str not in sys.path:
			sys.path.append(path_str)

logger = logging.getLogger(__name__)


def _eval_code(
	code: str,
	return_exec_globals: bool = False,
	exec_globals: Optional[Dict[str, Any]] = None,
) -> Any:
	try:
		exec_globals = {} if exec_globals is None else exec_globals
		exec(code, exec_globals)
		return exec_globals if return_exec_globals else "passed"
	except BaseException as exc:
		return f"failed: {exc}"


def _read_code(path: str) -> str:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Python world model not found: {path}")
	with open(path, "r", encoding="utf-8") as f:
		return f.read()


def _compile_functions(code_str: str) -> Tuple[Any, Any, Optional[str]]:
	exec_globals = _eval_code(code_str, return_exec_globals=True)
	if isinstance(exec_globals, str):
		return None, None, exec_globals

	init_state_func = exec_globals.get("init_state")
	predict_dynamics_func = exec_globals.get("predict_dynamics")

	if not callable(init_state_func):
		return None, None, "Function 'init_state' not found or not callable."
	if not callable(predict_dynamics_func):
		return None, None, "Function 'predict_dynamics' not found or not callable."

	return init_state_func, predict_dynamics_func, None


def _call_init_state(init_state_func: Any) -> Tuple[Any, Any, Optional[str]]:
	call_globals = {"init_state_func": init_state_func}
	result_or_error = _eval_code(
		"__result__ = init_state_func()",
		exec_globals=call_globals,
		return_exec_globals=True,
	)
	if isinstance(result_or_error, str):
		return None, None, result_or_error
	result = result_or_error.get("__result__")
	if not isinstance(result, tuple) or len(result) != 2:
		return None, None, "init_state() must return (state, hidden_state)."
	return result[0], result[1], None


def _call_predict(
	predict_dynamics_func: Any,
	state: Any,
	hidden_state: Any,
	action: Any,
) -> Tuple[Any, Any, Optional[str]]:
	call_globals = {
		"predict_dynamics_func": predict_dynamics_func,
		"state": state,
		"hidden_state": hidden_state,
		"action": action,
	}
	result_or_error = _eval_code(
		"__result__ = predict_dynamics_func(state, hidden_state, action)",
		exec_globals=call_globals,
		return_exec_globals=True,
	)
	if isinstance(result_or_error, str):
		return None, None, result_or_error
	result = result_or_error.get("__result__")
	if not isinstance(result, tuple) or len(result) != 2:
		return None, None, "predict_dynamics() must return (next_state, next_hidden_state)."
	return result[0], result[1], None


class PythonWorldModelTool:
	def __init__(
		self,
		env_name: str,
		use_obfuscated_code: bool = False,
		data_dir: Optional[str] = None,
		program_path: Optional[str] = None,
	) -> None:
		self.env_name = env_name
		self.use_obfuscated_code = use_obfuscated_code
		self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "example_benchmark")
		self.program_path = program_path or self._resolve_program_path()

		self._code_str: Optional[str] = None
		self._init_state_func: Optional[Any] = None
		self._predict_dynamics_func: Optional[Any] = None
		self._state: Any = None
		self._hidden_state: Any = None
		self._initialized = False

	def _resolve_program_path(self) -> str:
		programs_dir = "python_programs_obfuscated" if self.use_obfuscated_code else "python_programs"
		program_path = os.path.join(self.data_dir, programs_dir, f"{self.env_name}.py")
		return program_path

	def _ensure_loaded(self) -> None:
		if self._initialized:
			return

		self._code_str = _read_code(self.program_path)
		init_state_func, predict_dynamics_func, error = _compile_functions(self._code_str)
		if error:
			raise ToolException(error)

		state, hidden_state, error = _call_init_state(init_state_func)
		if error:
			raise ToolException(error)

		self._init_state_func = init_state_func
		self._predict_dynamics_func = predict_dynamics_func
		self._state = state
		self._hidden_state = hidden_state
		self._initialized = True

	def reset(self) -> None:
		self._initialized = False
		self._state = None
		self._hidden_state = None

	def step(self, action: Any) -> Dict[str, Any]:
		self._ensure_loaded()

		if self._predict_dynamics_func is None:
			raise ToolException("predict_dynamics() not available.")

		next_state, next_hidden_state, error = _call_predict(
			self._predict_dynamics_func,
			self._state,
			self._hidden_state,
			action,
		)
		if error:
			raise ToolException(error)

		self._state = next_state
		self._hidden_state = next_hidden_state

		return {
			"state": next_state,
			"hidden_state": next_hidden_state,
		}

	def get_langgraph_tools(self) -> List:
		"""Expose world model as a single step tool."""

		wm = self

		@tool(
			description=(
				"Apply an action to the Python world model and return a JSON string "
				"containing next state and next hidden state."
			),
		)
		def python_wm_step(action: Any) -> str:
			try:
				result = wm.step(action)
			except ToolException:
				raise
			except Exception as exc:
				logger.exception("python_wm_step failed")
				raise ToolException(str(exc)) from exc
			return json.dumps(result, ensure_ascii=False)

		return [python_wm_step]
