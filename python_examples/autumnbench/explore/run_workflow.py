"""
CLI entry for the new explore workflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

_FILE_DIR = Path(__file__).resolve().parent
_AUTUMNBENCH_DIR = _FILE_DIR.parent
_REPO_ROOT = _AUTUMNBENCH_DIR.parents[2]
for _p in [str(_AUTUMNBENCH_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_utils import get_llm  # noqa: E402
from workflow_graph import build_initial_state, create_workflow_graph  # noqa: E402


def run(
    env_name: str,
    data_dir: str,
    *,
    llm_model: str = "google/gemini-2.5-pro",
    max_explore_steps: int = 120,
    use_obfuscation: bool = False,
) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    llm = get_llm(model=llm_model)
    graph = create_workflow_graph(llm)
    init_state = build_initial_state(
        env_name=env_name,
        data_dir=data_dir,
        llm_model=llm_model,
        max_explore_steps=max_explore_steps,
        use_obfuscation=use_obfuscation,
    )
    final_state = graph.invoke(init_state)
    return {
        "code": final_state.get("code", ""),
        "trajectory_count": len(
            final_state.get("all_trajectories", [])
        ),
        "correct_trajectory_count": len(final_state.get("correct_trajectories", [])),
        "problem_count": len(final_state.get("problems") or []),
        "experiment_run_dir": final_state.get("experiment_run_dir", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run explore workflow graph")
    parser.add_argument("env_name", help="Environment id, e.g. 7XF97")
    parser.add_argument(
        "--data-dir",
        default=str(_AUTUMNBENCH_DIR / "example_benchmark"),
        help="Directory containing tests/programs .sexp files",
    )
    parser.add_argument("--llm-model", default="google/gemini-2.5-pro")
    parser.add_argument("--max-explore-steps", type=int, default=120)
    parser.add_argument("--use-obfuscation", action="store_true")
    parser.add_argument("--output", default=None, help="Optional path to save final code")
    args = parser.parse_args()

    result = run(
        env_name=args.env_name,
        data_dir=args.data_dir,
        llm_model=args.llm_model,
        max_explore_steps=args.max_explore_steps,
        use_obfuscation=args.use_obfuscation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output and result.get("code"):
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result["code"], encoding="utf-8")
        print(f"Saved code to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

