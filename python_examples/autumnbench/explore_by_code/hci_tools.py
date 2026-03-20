"""Human-collaboration tools for explicit phase declarations."""

from typing import List


def _format_traj_list(trajectory_paths: List[str]) -> str:
    if not trajectory_paths:
        return "(none)"
    return ", ".join(trajectory_paths)


def get_hci_tools():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "langchain_core is required to build tools. Install langchain-core first."
        ) from exc

    def declare_phase_fix_prediction(
        trajectory_paths: List[str],
        prediction_issue: str,
    ) -> str:
        traj_text = _format_traj_list(trajectory_paths)
        message = (
            "[Phase] fix_prediction | "
            f"trajectory_paths={traj_text} | prediction_issue={prediction_issue}"
        )
        print(message)
        return message

    def declare_phase_collect_data(
        trajectory_paths: List[str],
        prediction_issue: str,
    ) -> str:
        traj_text = _format_traj_list(trajectory_paths)
        message = (
            "[Phase] collect_data | "
            f"trajectory_paths={traj_text} | prediction_issue={prediction_issue}"
        )
        print(message)
        return message

    def declare_phase_explore_mechanism() -> str:
        message = "[Phase] explore_mechanism"
        print(message)
        return message

    return [
        StructuredTool.from_function(
            func=declare_phase_fix_prediction,
            name="declare_phase_fix_prediction",
            description=(
                "Declare that you are in the phase of modifying code to fix "
                "incorrect predictions on specific trajectories. Input: "
                "trajectory_paths, prediction_issue."
            ),
        ),
        StructuredTool.from_function(
            func=declare_phase_collect_data,
            name="declare_phase_collect_data",
            description=(
                "Declare that you are in the phase of collecting more data to "
                "help fix incorrect predictions on specific trajectories. "
                "Input: trajectory_paths, prediction_issue."
            ),
        ),
        StructuredTool.from_function(
            func=declare_phase_explore_mechanism,
            name="declare_phase_explore_mechanism",
            description=(
                "Declare that the current code already explains observed "
                "trajectories and you are now exploring for new mechanisms or "
                "new behaviors."
            ),
        ),
    ]


__all__ = ["get_hci_tools"]
