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
WORKSPACE_DIR = SCRIPT_DIR / "llm_workspace"
TRAJ_DIR = WORKSPACE_DIR / "traj"

for path in (str(MARA_ROOT), str(PY_EXAMPLES_DIR), str(AUTUMNBENCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

autumnstdlib = importlib.import_module("autumnbench.autumnstdlib").autumnstdlib
Interpreter = importlib.import_module("interpreter_module").Interpreter


class ResetRequest(BaseModel):
    env_name: str = Field(..., description="Environment name, e.g. 7XF97")
    data_dir: Optional[str] = Field(default=None, description="Directory containing programs/<ENV>.sexp")
    seed: int = 0


class StepRequest(BaseModel):
    action: str


class SaveTrajectoryRequest(BaseModel):
    filename: Optional[str] = Field(default=None, description="Optional base filename without path")


class EnvSession:
    def __init__(self) -> None:
        self._lock = Lock()
        self.interpreter: Optional[Any] = None
        self.env_name: Optional[str] = None
        self.data_dir: Optional[Path] = None
        self.seed: Optional[int] = None
        self.current_state: Optional[Dict[str, Any]] = None
        self.actions: List[str] = []
        self.transitions: List[Dict[str, Any]] = []

    def _load_program(self, env_name: str, data_dir: Path) -> str:
        program_path = data_dir / "programs" / f"{env_name}.sexp"
        if not program_path.is_file():
            raise FileNotFoundError(f"Program not found: {program_path}")
        return program_path.read_text(encoding="utf-8")

    def _render_state(self) -> Dict[str, Any]:
        if self.interpreter is None:
            raise RuntimeError("Environment not initialized. Call reset first.")
        return json.loads(self.interpreter.render_all())

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

    def reset(self, env_name: str, data_dir: Optional[str], seed: int) -> Dict[str, Any]:
        with self._lock:
            resolved_data_dir = Path(data_dir).resolve() if data_dir else (AUTUMNBENCH_DIR / "example_benchmark").resolve()
            program = self._load_program(env_name, resolved_data_dir)

            interpreter = Interpreter()
            interpreter.run_script(program, autumnstdlib, "", seed)

            self.interpreter = interpreter
            self.env_name = env_name
            self.data_dir = resolved_data_dir
            self.seed = seed
            self.current_state = self._render_state()
            self.actions = []
            self.transitions = []

            return {
                "ok": True,
                "env": env_name,
                "seed": seed,
                "data_dir": str(resolved_data_dir),
                "state": self.current_state,
            }

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

            return {
                "ok": True,
                "step": len(self.transitions),
                "action": action,
                "state": next_state,
            }

    def save_trajectory(self, filename: Optional[str]) -> Dict[str, Any]:
        with self._lock:
            if self.env_name is None or self.data_dir is None:
                raise RuntimeError("Environment not initialized. Call reset first.")

            env_dir = TRAJ_DIR / self.env_name
            env_dir.mkdir(parents=True, exist_ok=True)

            if filename:
                safe_name = os.path.basename(filename)
                if not safe_name.endswith(".json"):
                    safe_name = f"{safe_name}.json"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = f"trajectory_{timestamp}.json"

            out_path = env_dir / safe_name

            payload = {
                "env": self.env_name,
                "seed": self.seed,
                "data_dir": str(self.data_dir),
                "actions": list(self.actions),
                "num_transitions": len(self.transitions),
                "trajectory": list(self.transitions),
            }

            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            return {
                "ok": True,
                "path": str(out_path),
                "num_transitions": len(self.transitions),
            }


app = FastAPI(title="AutumnBench Environment API", version="1.0.0")
session = EnvSession()


@app.get("/health")
def health() -> Dict[str, Any]:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    return {"ok": True}


@app.post("/reset")
def reset_env(payload: ResetRequest) -> Dict[str, Any]:
    try:
        return session.reset(payload.env_name, payload.data_dir, payload.seed)
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
        return session.save_trajectory(payload.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
