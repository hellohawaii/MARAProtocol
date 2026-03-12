import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


SCRIPT_DIR = Path(__file__).resolve().parent
AUTUMNBENCH_DIR = SCRIPT_DIR.parent
PY_EXAMPLES_DIR = AUTUMNBENCH_DIR.parent
MARA_ROOT = AUTUMNBENCH_DIR.parent.parent
DEFAULT_WORKSPACE_DIR = (SCRIPT_DIR / "llm_workspace").resolve()
RUNS_WORKSPACE_ROOT_DIR = (SCRIPT_DIR / "llm_workspace_runs").resolve()

for path in (str(MARA_ROOT), str(PY_EXAMPLES_DIR), str(AUTUMNBENCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from .log_utils import ensure_logs_root_dir
    from .traj_visualization_utils import save_trajectory_visualization
except ImportError:  # pragma: no cover
    from log_utils import ensure_logs_root_dir
    from traj_visualization_utils import save_trajectory_visualization

autumnstdlib = importlib.import_module("autumnbench.autumnstdlib").autumnstdlib
Interpreter = importlib.import_module("interpreter_module").Interpreter


def _load_obfuscation_mapping(data_dir: Path) -> Dict[str, str]:
    mapping_path = data_dir / "obfuscation_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Obfuscation mapping not found at {mapping_path}")
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def _obfuscate_scene_graph(render_dict: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    obfuscated: Dict[str, Any] = {}
    for key, value in render_dict.items():
        if key == "GRID_SIZE":
            obfuscated[key] = value
            continue
        if key not in mapping:
            raise KeyError(f"Missing obfuscation mapping for object: {key}")
        obfuscated[mapping[key]] = value
    return obfuscated


class ResetRequest(BaseModel):
    pass


class StepRequest(BaseModel):
    action: str


class SaveTrajectoryRequest(BaseModel):
    filename: Optional[str] = Field(default=None, description="Optional base filename without path")
    dir: Optional[str] = Field(default=None, description="Optional directory name under workspace_dir")


class SetWorkspaceRequest(BaseModel):
    workspace_dir: str = Field(..., description="Absolute workspace directory for current run")
    run_id: Optional[str] = Field(default=None, description="Optional run identifier for logging")


class SetEnvNameRequest(BaseModel):
    env_name: str = Field(..., description="Environment name, e.g. 7XF97")


class EnvSession:
    def __init__(self) -> None:
        self._lock = Lock()
        self.workspace_dir: Path = DEFAULT_WORKSPACE_DIR
        self.current_run_id: Optional[str] = None
        self.configured_env_name: Optional[str] = None
        self.interpreter: Optional[Any] = None
        self.env_name: Optional[str] = None
        self.data_dir: Optional[Path] = None
        self.seed: Optional[int] = None
        self._obfuscation_mapping: Optional[Dict[str, str]] = None
        self.current_state: Optional[Dict[str, Any]] = None
        self.actions: List[str] = []
        self.transitions: List[Dict[str, Any]] = []
        self._ensure_workspace_dirs()

    def _ensure_workspace_dirs(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (self.workspace_dir / "traj").mkdir(parents=True, exist_ok=True)
        ensure_logs_root_dir()

    def _validate_workspace_dir(self, workspace_dir: Path) -> None:
        if workspace_dir == DEFAULT_WORKSPACE_DIR:
            return
        if workspace_dir.is_relative_to(RUNS_WORKSPACE_ROOT_DIR):
            return
        raise ValueError(
            f"Workspace must be under {RUNS_WORKSPACE_ROOT_DIR} or equal to {DEFAULT_WORKSPACE_DIR}, got: {workspace_dir}"
        )

    def set_workspace(self, workspace_dir: str, run_id: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            resolved = Path(workspace_dir).resolve()
            self._validate_workspace_dir(resolved)
            resolved.mkdir(parents=True, exist_ok=True)
            (resolved / "traj").mkdir(parents=True, exist_ok=True)
            self.workspace_dir = resolved
            self.current_run_id = run_id
            return {
                "ok": True,
                "workspace_dir": str(self.workspace_dir),
                "run_id": self.current_run_id,
            }

    def set_env_name(self, env_name: str) -> Dict[str, Any]:
        with self._lock:
            self.configured_env_name = env_name
            return {"ok": True}

    def _load_program(self, env_name: str, data_dir: Path) -> str:
        program_path = data_dir / "programs" / f"{env_name}.sexp"
        if not program_path.is_file():
            raise FileNotFoundError(f"Program not found: {program_path}")
        return program_path.read_text(encoding="utf-8")

    def _render_state(self) -> Dict[str, Any]:
        if self.interpreter is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        state = json.loads(self.interpreter.render_all())
        if self._obfuscation_mapping is not None:
            state = _obfuscate_scene_graph(state, self._obfuscation_mapping)
        return state

    def _apply_action(self, action: str) -> bool:
        if self.interpreter is None:
            raise RuntimeError("Environment not initialized. Call reset first.")

        if action == "left":
            self.interpreter.left()
            return True
        if action == "right":
            self.interpreter.right()
            return True
        if action == "up":
            self.interpreter.up()
            return True
        if action == "down":
            self.interpreter.down()
            return True
        if action == "noop":
            return True
        if action.startswith("click"):
            parts = action.split()
            if len(parts) < 3:
                return False
            try:
                x = int(parts[1])
                y = int(parts[2])
            except ValueError:
                return False
            self.interpreter.click(x, y)
            return True
        return False

    def reset(self) -> Dict[str, Any]:
        with self._lock:
            env_name = self.configured_env_name
            if not env_name:
                raise RuntimeError("Environment not configured. Internal setup must call /_set_env_name first.")
            resolved_data_dir = (AUTUMNBENCH_DIR / "example_benchmark").resolve()
            program = self._load_program(env_name, resolved_data_dir)
            seed = 0

            interpreter = Interpreter()
            interpreter.run_script(program, autumnstdlib, "", seed)

            self.interpreter = interpreter
            self.env_name = env_name
            self.data_dir = resolved_data_dir
            self.seed = seed
            
            obfuscated = os.getenv("OBFUSCATED", "0") == "1"
            self._obfuscation_mapping = _load_obfuscation_mapping(resolved_data_dir) if obfuscated else None
            self.current_state = self._render_state()
            self.actions = []
            self.transitions = []

            return self.current_state

    def step(self, action: str) -> Dict[str, Any]:
        with self._lock:
            if self.interpreter is None or self.current_state is None:
                raise RuntimeError("Environment not initialized. Call reset first.")

            valid = self._apply_action(action)
            if not valid:
                raise ValueError(f"Unsupported action: {action}")

            self.interpreter.step()
            next_state = self._render_state()

            self.transitions.append(
                {
                    "step": len(self.transitions) + 1,
                    "state": self.current_state,
                    "action": action,
                    "new_state": next_state,
                }
            )
            self.actions.append(action)
            self.current_state = next_state

            return next_state

    def save_trajectory(self, filename: Optional[str], dir: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if self.env_name is None or self.data_dir is None:
                raise RuntimeError("Environment not initialized. Call reset first.")

            target_dir_name = dir if dir else "traj"
            traj_dir = self.workspace_dir / target_dir_name
            traj_dir.mkdir(parents=True, exist_ok=True)

            if filename:
                safe_name = os.path.basename(filename)
                if not safe_name.endswith(".json"):
                    safe_name = f"{safe_name}.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = f"trajectory_{timestamp}.json"

            out_path = traj_dir / safe_name

            payload = {
                "env": self.env_name,
                "seed": self.seed,
                "data_dir": str(self.data_dir),
                "actions": list(self.actions),
                "num_transitions": len(self.transitions),
                "trajectory": list(self.transitions),
            }

            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            try:
                save_trajectory_visualization(
                    payload=payload,
                    out_path=out_path,
                    run_id=self.current_run_id,
                )
            except Exception:
                # Fail-open: visualization errors should not affect trajectory saving.
                pass
            return {"success": True}


app = FastAPI(title="AutumnBench Environment API", version="1.0.0")
session = EnvSession()


@app.get("/health")
def health() -> Dict[str, Any]:
    session._ensure_workspace_dirs()
    return {
        "ok": True,
        "workspace_dir": str(session.workspace_dir),
        "run_id": session.current_run_id,
    }


@app.post("/set_workspace")
def set_workspace(payload: SetWorkspaceRequest) -> Dict[str, Any]:
    try:
        return session.set_workspace(payload.workspace_dir, payload.run_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/_set_env_name")
def set_env_name(payload: SetEnvNameRequest) -> Dict[str, Any]:
    try:
        return session.set_env_name(payload.env_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset")
def reset_env(payload: ResetRequest) -> Dict[str, Any]:
    try:
        return session.reset()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step_env(payload: StepRequest) -> Dict[str, Any]:
    try:
        return session.step(payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/save_trajectory")
def save_trajectory(payload: SaveTrajectoryRequest) -> Dict[str, Any]:
    try:
        return session.save_trajectory(payload.filename, payload.dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
