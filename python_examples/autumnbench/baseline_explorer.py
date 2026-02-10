"""
Baseline Explorer Agent — iterative explore-then-refine agent for AutumnBench.

This agent alternates between two subgraphs:
1. **Exploration** (``explore_graph.py``) — a ReAct agent interacts with the
   environment, collecting trajectories, then a summarise node produces
   structured Q&A pairs.
2. **Code Refinement** (``code_refine_graph.py``) — generates or iteratively
   refines a ``predict_dynamics`` function to match all collected trajectories.

Across iterations the agent maintains:
- A trajectory library (all trajectories collected so far)
- The current ``predict_dynamics`` source code
- Structured Q&A knowledge from exploration
- Pending diagnostic questions (from the refine LLM when it cannot reach 100%)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parents[2]  # /app
for _p in [str(_REPO_ROOT), str(_FILE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from code_refine_graph import RefineState, create_refine_graph  # noqa: E402
from explore_graph import ExploreState, create_explore_graph  # noqa: E402
from langchain_utils import get_llm  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================
# Main State
# ============================================================

class MainState(TypedDict):
    current_code: Optional[str]
    trajectory_library: list            # List of trajectories in executor format
    new_trajectories: list              # Trajectories from the latest exploration
    explored_qa: list                   # List[dict] with {"question": str, "answer": str}
    new_qa: list                        # Q&A pairs from the latest exploration round
    pending_questions: list             # Questions from refine LLM for explorer (List[str])
    error_frames_for_explorer: list     # Mispredicted frames for explorer context
    code_perfect: bool                  # Whether last refine achieved 100%
    questions_from_refine: bool         # True if last exploration used questions from refine
    exploration_sufficient: bool        # True if explore LLM decides no more exploration needed
    iteration: int
    max_iterations: int
    max_refine_iterations: int
    max_explore_steps: int
    env_name: str
    data_dir: str
    llm_model: str
    use_obfuscation: bool               # Whether to obfuscate object-type names


# ============================================================
# Graph Nodes
# ============================================================

def _create_graph_nodes(llm):
    """Return node functions as closures over the *llm* instance."""

    explore_subgraph = create_explore_graph(llm)
    refine_subgraph = create_refine_graph(llm)

    # ----------------------------------------------------------
    # Explore node — unified exploration (free or targeted)
    #
    # The explore subgraph handles both cases internally:
    #   - If pending_questions is non-empty (from refine), it uses them
    #     directly and sets questions_from_refine=True.
    #   - If pending_questions is empty, it generates its own questions
    #     first and sets questions_from_refine=False.
    # ----------------------------------------------------------
    def explore_node(state: MainState) -> dict:
        logger.info(
            "=== Exploration (iteration %d) ===", state["iteration"],
        )

        explore_input: Dict[str, Any] = {
            "current_code": state.get("current_code"),
            "explored_qa": state.get("explored_qa", []),
            "pending_questions": state.get("pending_questions", []),
            "error_frames": state.get("error_frames_for_explorer", []),
            "max_explore_steps": state.get("max_explore_steps", 200),
            "env_name": state["env_name"],
            "data_dir": state["data_dir"],
            "use_obfuscation": state.get("use_obfuscation", False),
            "agent_messages": [],
            "new_trajectories": [],
            "new_qa": [],
            "questions_from_refine": False,
            "exploration_sufficient": False,
        }

        explore_result = explore_subgraph.invoke(explore_input)

        # Merge new trajectories into library
        all_trajs = list(state.get("trajectory_library", []))
        new_trajs = explore_result.get("new_trajectories", [])
        all_trajs.extend(new_trajs)

        # Merge new Q&A into historical Q&A
        all_qa = list(state.get("explored_qa", []))
        new_qa = explore_result.get("new_qa", [])
        all_qa.extend(new_qa)

        questions_from_refine = explore_result.get("questions_from_refine", False)
        exploration_sufficient = explore_result.get("exploration_sufficient", False)

        if exploration_sufficient:
            logger.info(
                "Exploration subgraph signalled exploration_sufficient — "
                "will skip refinement and exit main loop."
            )

        mode = "targeted" if questions_from_refine else "free"
        logger.info(
            "%s exploration collected %d new trajectories (%d total), %d Q&A pairs",
            mode.capitalize(), len(new_trajs), len(all_trajs), len(new_qa),
        )

        return {
            "trajectory_library": all_trajs,
            "new_trajectories": new_trajs,
            "explored_qa": all_qa,
            "new_qa": new_qa,
            "pending_questions": [],  # Clear — they've been addressed
            "questions_from_refine": questions_from_refine,
            "exploration_sufficient": exploration_sufficient,
        }

    # ----------------------------------------------------------
    # Refine node — invokes the refinement subgraph
    # ----------------------------------------------------------
    def refine_code_node(state: MainState) -> dict:
        logger.info(
            "=== Code refinement phase (iteration %d) ===", state["iteration"]
        )

        was_targeted = bool(state.get("questions_from_refine", False))
        has_existing_code = state.get("current_code") is not None

        # Common fields shared by both branches
        common: Dict[str, Any] = {
            "all_trajectories": state["trajectory_library"],
            "conversation_history": [],  # fresh conversation each main-loop iteration
            "refine_iteration": 0,
            "max_refine_iterations": state.get("max_refine_iterations", 5),
            "accuracy": 0.0,
            "total_correct": 0,
            "total_frames": 0,
            "execution_error": None,
            "per_trajectory_results": [],
            "first_error_frames": [],
            "questions": [],
            "code_perfect": False,
            "new_qa": state.get("new_qa", []),
            "all_qa": state.get("explored_qa", []),
        }

        if was_targeted and has_existing_code:
            # After targeted exploration: present findings (current code +
            # new trajectories/Q&A) so the LLM can refine from there.
            refine_input: Dict[str, Any] = {
                **common,
                "code": state.get("current_code") or "",
                "targeted_trajectories": state.get("new_trajectories", []),
                "from_targeted_exploration": True,
                "is_first_generation": False,
            }
        else:
            # After free exploration (or first iteration): generate code
            # from scratch using all trajectories and Q&A knowledge.
            refine_input = {
                **common,
                "code": "",
                "targeted_trajectories": [],
                "from_targeted_exploration": False,
                "is_first_generation": True,
            }

        refine_result = refine_subgraph.invoke(refine_input)

        return {
            "current_code": refine_result["code"],
            "code_perfect": refine_result["code_perfect"],
            "pending_questions": refine_result.get("questions", []),
            "error_frames_for_explorer": refine_result.get("first_error_frames", []),
            "iteration": state["iteration"] + 1,
        }

    return explore_node, refine_code_node


# ============================================================
# Routing
# ============================================================

def route_after_explore(state: MainState) -> str:
    """Route after exploration: skip refinement if exploration is deemed sufficient."""
    if state.get("exploration_sufficient"):
        logger.info("Exploration sufficient — skipping refinement, ending main loop.")
        return END
    return "refine"


def decide_next(state: MainState) -> str:
    """Route after refinement: stop or explore again.

    The explore subgraph handles both free and targeted exploration
    internally based on whether ``pending_questions`` is populated.
    """
    if state["iteration"] >= state["max_iterations"]:
        logger.info("Max iterations (%d) reached — stopping.", state["max_iterations"])
        return END
    if state.get("pending_questions"):
        logger.info("Investigation questions pending — next exploration will be targeted.")
    elif state.get("code_perfect"):
        logger.info("Code is perfect — next exploration will be free (for more coverage).")
    return "explore"


# ============================================================
# Graph Builder
# ============================================================

def create_main_graph(llm):
    """Build and compile the main orchestration graph.

    Flow::

        START ──▶ explore ──▶ route_after_explore
                     ▲              │
                     │    ┌─────────┴──────────┐
                     │    ▼                     ▼
                     │  refine ──▶ decide      END
                     │               │      (exploration_sufficient)
                     │    ┌──────────┴─────┐
                     │    ▼                ▼
                     └─ explore            END
    """
    explore_node, refine_code_node = _create_graph_nodes(llm)

    workflow = StateGraph(MainState)

    workflow.add_node("explore", explore_node)
    workflow.add_node("refine", refine_code_node)

    workflow.set_entry_point("explore")
    workflow.add_conditional_edges("explore", route_after_explore, {
        "refine": "refine",
        END: END,
    })
    workflow.add_conditional_edges("refine", decide_next, {
        "explore": "explore",
        END: END,
    })

    return workflow.compile()


# ============================================================
# Entry Point
# ============================================================

def run(
    env_name: str,
    data_dir: str,
    *,
    max_iterations: int = 5,
    max_refine_iterations: int = 5,
    max_explore_steps: int = 200,
    llm_model: str = "google/gemini-2.5-pro",
    use_obfuscation: bool = False,
) -> Dict[str, Any]:
    """Run the baseline explorer agent.

    Args:
        env_name: Name of the environment (.sexp file without extension).
        data_dir: Directory containing ``tests/{env_name}.sexp``.
        max_iterations: Maximum explore-refine cycles.
        max_refine_iterations: Maximum LLM refinement calls per cycle.
        max_explore_steps: Maximum actions per exploration phase.
        llm_model: Model identifier for OpenRouter.
        use_obfuscation: If True, obfuscate object-type names in observations
            and trajectories using ``obfuscation_mapping.json``.

    Returns:
        A dict with ``code``, ``trajectory_library``, ``explored_qa``,
        ``code_perfect``, and ``iterations``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    llm = get_llm(model=llm_model)
    graph = create_main_graph(llm)

    initial_state: Dict[str, Any] = {
        "current_code": None,
        "trajectory_library": [],
        "new_trajectories": [],
        "explored_qa": [],
        "new_qa": [],
        "pending_questions": [],
        "error_frames_for_explorer": [],
        "code_perfect": False,
        "exploration_sufficient": False,
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_refine_iterations": max_refine_iterations,
        "max_explore_steps": max_explore_steps,
        "env_name": env_name,
        "data_dir": data_dir,
        "llm_model": llm_model,
        "use_obfuscation": use_obfuscation,
        "questions_from_refine": False,
    }

    final_state = graph.invoke(initial_state)

    result = {
        "code": final_state.get("current_code"),
        "trajectory_library": final_state.get("trajectory_library", []),
        "explored_qa": final_state.get("explored_qa", []),
        "code_perfect": final_state.get("code_perfect", False),
        "iterations": final_state.get("iteration", 0),
    }

    logger.info("=== Baseline Explorer finished ===")
    logger.info("Iterations: %d", result["iterations"])
    logger.info("Code perfect: %s", result["code_perfect"])
    logger.info("Trajectories collected: %d", len(result["trajectory_library"]))
    logger.info("Q&A pairs: %d", len(result["explored_qa"]))

    if result["code"]:
        logger.info("Final code:\n%s", result["code"])

    return result


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline Explorer Agent for AutumnBench"
    )
    parser.add_argument("env_name", help="Environment name (without .sexp)")
    parser.add_argument(
        "--data-dir",
        default=str(_FILE_DIR / "example_benchmark"),
        help="Directory containing tests/ with .sexp files",
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-refine-iterations", type=int, default=5)
    parser.add_argument("--max-explore-steps", type=int, default=200)
    parser.add_argument("--llm-model", default="google/gemini-2.5-pro")
    parser.add_argument(
        "--use-obfuscation",
        action="store_true",
        default=False,
        help="Obfuscate object-type names using obfuscation_mapping.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the final predict_dynamics code",
    )

    args = parser.parse_args()

    result = run(
        env_name=args.env_name,
        data_dir=args.data_dir,
        max_iterations=args.max_iterations,
        max_refine_iterations=args.max_refine_iterations,
        max_explore_steps=args.max_explore_steps,
        llm_model=args.llm_model,
        use_obfuscation=args.use_obfuscation,
    )

    if args.output and result["code"]:
        with open(args.output, "w") as f:
            f.write(result["code"])
        print(f"Code written to {args.output}")
