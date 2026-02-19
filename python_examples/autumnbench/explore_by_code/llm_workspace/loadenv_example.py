import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
AUTUMNBENCH_DIR = SCRIPT_DIR.parent
PY_EXAMPLES_DIR = AUTUMNBENCH_DIR.parent
MARA_ROOT = AUTUMNBENCH_DIR.parent.parent

for path in (str(MARA_ROOT), str(PY_EXAMPLES_DIR), str(AUTUMNBENCH_DIR)):
	if path not in sys.path:
		sys.path.insert(0, path)

autumnstdlib = importlib.import_module("autumnbench.autumnstdlib").autumnstdlib
Interpreter = importlib.import_module("interpreter_module").Interpreter


def _load_program(env_name: str, data_dir: Path) -> str:
	program_path = data_dir / "programs" / f"{env_name}.sexp"
	if not program_path.is_file():
		raise FileNotFoundError(f"Program not found: {program_path}")
	return program_path.read_text(encoding="utf-8")


def _render_state(interpreter: Any) -> Dict[str, Any]:
	return json.loads(interpreter.render_all())


def _build_sample_actions(grid_size: int) -> List[str]:
	if grid_size <= 0:
		return ["noop", "noop"]

	max_idx = grid_size - 1
	mid = grid_size // 2
	return [
		"noop",
		"left",
		"right",
		"up",
		"down",
		f"click 0 0",
		f"click {mid} {mid}",
		f"click {max_idx} {max_idx}",
		"noop",
	]


def _apply_action(interpreter: Any, action: str) -> bool:
	if action == "left":
		interpreter.left()
		return True
	if action == "right":
		interpreter.right()
		return True
	if action == "up":
		interpreter.up()
		return True
	if action == "down":
		interpreter.down()
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
		interpreter.click(x, y)
		return True
	return False


def run_minimal_example(env_name: str, data_dir: Path, out_root: Path, seed: int) -> Path:
	prog = _load_program(env_name, data_dir)

	interpreter = Interpreter()
	interpreter.run_script(prog, autumnstdlib, "", seed)

	initial_state = _render_state(interpreter)
	grid_size = int(initial_state.get("GRID_SIZE", 0))
	actions = _build_sample_actions(grid_size)

	transitions: List[Dict[str, Any]] = []
	current_state = initial_state

	for step_idx, action in enumerate(actions, start=1):
		valid = _apply_action(interpreter, action)
		if not valid:
			continue
		interpreter.step()
		next_state = _render_state(interpreter)
		transitions.append(
			{
				"step": step_idx,
				"state": current_state,
				"action": action,
				"new_state": next_state,
			}
		)
		current_state = next_state

	env_out_dir = out_root / env_name
	env_out_dir.mkdir(parents=True, exist_ok=True)

	payload = {
		"env": env_name,
		"seed": seed,
		"data_dir": str(data_dir.resolve()),
		"actions": actions,
		"num_transitions": len(transitions),
		"trajectory": transitions,
	}

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_path = env_out_dir / f"trajectory_{timestamp}.json"
	out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Minimal example: load ENV.sexp, interact with sample actions, and save trajectory.",
	)
	parser.add_argument("env", help="Environment name, e.g. 7XF97")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=AUTUMNBENCH_DIR / "example_benchmark",
		help="Directory containing programs/<ENV>.sexp",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=SCRIPT_DIR / "traj",
		help="Trajectory output root directory",
	)
	parser.add_argument("--seed", type=int, default=0)
	args = parser.parse_args()

	out_path = run_minimal_example(
		env_name=args.env,
		data_dir=args.data_dir,
		out_root=args.out_dir,
		seed=args.seed,
	)
	print(f"Saved trajectory to: {out_path}")


if __name__ == "__main__":
	main()
