"""
Global controller policy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from error_analysis_graph import run_error_analysis
from shared_runtime import ProblemItem


def _pick_target_problem(problems: List[ProblemItem]) -> Optional[ProblemItem]:
    for p in problems:
        if bool(p.get("solved", False)):
            continue
        if bool(p.get("tried", False)):
            continue
        return dict(p)
    return None


def choose_latest_strategy(llm, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Baseline policy:
    1) always choose the latest generated code;
    2) evaluate latest code against trajectory library;
    3) choose one repair target problem for this code.
    """
    latest_code = state.get("code", "")
    all_trajectories = list(state.get("all_trajectories", []))
    known_problems = state.get("problems")
    evaluate_out: Dict[str, Any] = {}
    if not latest_code.strip():
        problems = [
            {
                "problem_id": "initial_generation",
                "summary": "No code exists yet. Need to write the initial candidate_model.py.",
                "problem_samples": [],
                "tried": False,
                "solved": False,
            }
        ]
    elif known_problems is None:
        evaluate_out = run_error_analysis(
            llm,
            code=latest_code.strip(),
            trajectories=all_trajectories,
            problems=[],
            target_problem=None,
        )
        problems = list(evaluate_out.get("problems", []))
    else:
        problems = list(known_problems)
    target_problem = _pick_target_problem(problems)
    return {
        "code": latest_code,
        "all_trajectories": all_trajectories,
        "correct_trajectories": evaluate_out.get(
            "correct_trajectories", state.get("correct_trajectories", [])
        ),
        "problems": problems,
        "coding_can_explain": evaluate_out.get(
            "coding_can_explain", state.get("coding_can_explain", False)
        ),
        "error_samples": evaluate_out.get("error_samples", state.get("error_samples", [])),
        "target_problem": target_problem,
        "target_problem_fixed": False,
        "should_terminate_workflow": target_problem is None and len(problems) > 0,
    }

