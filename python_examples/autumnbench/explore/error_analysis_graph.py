"""
Unified graph: run code, collect first errors, classify errors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from code_refine_graph import execute_code_on_trajectories, find_first_error_frames
from shared_runtime import ProblemItem, TrajectoryRecord, build_problem_item


class TargetKeepIndices(BaseModel):
    keep_indices: List[int] = Field(default_factory=list)


class ErrorLabels(BaseModel):
    labels: List[str] = Field(default_factory=list)


class ErrorAnalysisState(TypedDict):
    code: str
    trajectories: List[TrajectoryRecord]
    correct_trajectories: List[TrajectoryRecord]
    problems: List[ProblemItem]
    target_problem: Optional[ProblemItem]
    error_samples: list
    coding_can_explain: bool
    target_problem_errors: list


def create_error_analysis_graph(llm):
    def _strip_traj_idx(err: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in dict(err).items() if k != "traj_idx"}

    def execute_and_collect_node(state: ErrorAnalysisState) -> Dict[str, Any]:
        code = (state.get("code") or "").strip()
        trajs = list(state.get("trajectories", []))
        if not code:
            return {
                "coding_can_explain": False,
                "error_samples": [{"error": "missing code"}],
                "target_problem_errors": [],
                "correct_trajectories": [],
            }
        if not trajs:
            return {
                "coding_can_explain": True,
                "error_samples": [],
                "target_problem_errors": [],
                "correct_trajectories": [],
            }
        result = execute_code_on_trajectories(code, trajs)
        if not result.get("success"):
            return {
                "coding_can_explain": False,
                "error_samples": [{"error": result.get("error_info", {}).get("error_message", "execution failed")}],
                "target_problem_errors": [],
                "correct_trajectories": [],
            }
        errors = find_first_error_frames(result.get("results", []), trajs)
        error_traj_indices = set()
        for err in errors:
            idx = err.get("traj_idx")
            if isinstance(idx, int) and 0 <= idx < len(trajs):
                error_traj_indices.add(idx)
        correct_trajectories = [
            traj
            for i, traj in enumerate(trajs)
            if i not in error_traj_indices
        ]
        return {
            "coding_can_explain": len(errors) == 0,
            "error_samples": errors,
            "target_problem_errors": [],
            "correct_trajectories": correct_trajectories,
        }

    def classify_node(state: ErrorAnalysisState) -> Dict[str, Any]:
        errors = list(state.get("error_samples", []))
        problems: List[ProblemItem] = list(state.get("problems", []))
        target_problem = state.get("target_problem")
        if not errors:
            if target_problem:
                return {"problems": problems, "target_problem_errors": []}
            return {
                "problems": [],
                "target_problem_errors": [],
                "correct_trajectories": list(state.get("trajectories", [])),
            }

        # Target mode: filter errors that belong to target problem P.
        code_str = state.get("code", "")
        env_basics = """Environment basics:
- Deterministic GRID_SIZE x GRID_SIZE world.
- You can treat this as an MDP/POMDP-style dynamics problem: visible observations may
  be sufficient in some environments, while others require hidden_state to represent
  latent dynamics.
- Valid actions for env.step(action):
  - click x y: Click on the cell at location (x, y). For GRID_SIZE, x and y
    must each be between 0 and GRID_SIZE-1 inclusive. (Do NOT use commas, e.g., "click 3, 4" is invalid).
  - left: Press the left arrow key.
  - right: Press the right arrow key.
  - up: Press the up arrow key.
  - down: Press the down arrow key.
  - noop: Do nothing and continue to the next step."""

        if target_problem:
            target_summary = str(target_problem.get("summary", ""))
            keep_indices: List[int] = []
            
            system_prompt = f"""You are an expert AI assistant helping to analyze code execution errors.
{env_basics}

Your task is to filter the given list of errors and identify which ones belong to the target problem P.
Output the indices (keep_indices) of the errors that match the target problem P."""

            user_prompt = f"""Target Problem P: {target_summary}

Current Code:
```python
{code_str}
```

Error List:
{errors}"""

            try:
                structured = llm.with_structured_output(TargetKeepIndices)
                out: TargetKeepIndices = structured.invoke([
                    ("system", system_prompt),
                    ("user", user_prompt)
                ])
                keep_indices = out.keep_indices
            except Exception:
                keep_indices = list(range(len(errors)))
            filtered: List[Dict[str, Any]] = []
            for i in keep_indices:
                if isinstance(i, int) and 0 <= i < len(errors):
                    filtered.append(errors[i])
            return {
                "problems": problems,
                "target_problem_errors": filtered,
            }

        # General mode: classify each error into a label and update problem list.
        labels: List[str] = []
        trajectories = list(state.get("trajectories", []))
        
        system_prompt = f"""You are an expert AI assistant helping to analyze code execution errors.
{env_basics}

Your task is to determine which of the prediction errors are of the same type (i.e., highly likely to originate from the same defect in the code).
Classify these errors and provide a detailed description for each category (explaining the specific manifestation of the error and the possible code cause).
This description will be used later to group errors, guide code modifications, and find similar errors.

Output a list of `labels`, the length of which MUST exactly match the number of errors.
For errors belonging to the same category, you MUST output the exact same detailed description string in the list."""

        user_prompt = f"""Current Code:
```python
{code_str}
```

Error List:
{errors}"""

        try:
            structured = llm.with_structured_output(ErrorLabels)
            out: ErrorLabels = structured.invoke([
                ("system", system_prompt),
                ("user", user_prompt)
            ])
            labels = out.labels
        except Exception:
            labels = ["generic_error"] * len(errors)
        if len(labels) != len(errors):
            labels = (labels + ["generic_error"] * len(errors))[:len(errors)]

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for label, err in zip(labels, errors):
            traj_idx = err.get("traj_idx")
            if not isinstance(traj_idx, int) or not (0 <= traj_idx < len(trajectories)):
                continue
            grouped[str(label).strip() or "generic_error"].append(
                {
                    "trajectory": trajectories[traj_idx],
                    "error": _strip_traj_idx(err),
                }
            )

        rebuilt_problems: List[ProblemItem] = []
        for i, (label, group_samples) in enumerate(grouped.items(), start=1):
            rebuilt_problems.append(
                {
                    "problem_id": f"problem_{i}",
                    "summary": label,
                    "problem_samples": group_samples,
                    "tried": False,
                    "solved": False,
                }
            )
        return {"problems": rebuilt_problems, "target_problem_errors": []}

    workflow = StateGraph(ErrorAnalysisState)
    workflow.add_node("execute_and_collect", execute_and_collect_node)
    workflow.add_node("classify", classify_node)
    workflow.set_entry_point("execute_and_collect")
    workflow.add_edge("execute_and_collect", "classify")
    workflow.add_edge("classify", END)
    return workflow.compile()


def run_error_analysis(
    llm,
    *,
    code: str,
    trajectories: List[TrajectoryRecord],
    problems: List[ProblemItem],
    target_problem: Optional[ProblemItem] = None,
) -> Dict[str, Any]:
    graph = create_error_analysis_graph(llm)
    out = graph.invoke(
        {
            "code": code,
            "trajectories": trajectories,
            "problems": problems,
            "target_problem": target_problem,
            "correct_trajectories": [],
            "error_samples": [],
            "coding_can_explain": False,
            "target_problem_errors": [],
        }
    )
    return {
        "coding_can_explain": bool(out.get("coding_can_explain", False)),
        "error_samples": out.get("error_samples", []),
        "problems": out.get("problems", problems),
        "target_problem_errors": out.get("target_problem_errors", []),
        "correct_trajectories": out.get("correct_trajectories", []),
    }

