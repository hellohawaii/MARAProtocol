"""Human-collaboration tools for explicit phase declarations and model-initiated human interaction."""

from typing import Any, Callable, Coroutine, List, Optional


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

    def declare_phase_run_trial() -> str:
        message = "[Phase] run_trial"
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
        StructuredTool.from_function(
            func=declare_phase_run_trial,
            name="declare_phase_run_trial",
            description=(
                "Declare that you have built sufficient understanding of the "
                "environment's world model to formulate a goal-reaching plan, "
                "and are now starting a trial attempt to achieve the goal. "
                "Use this only when you are ready to commit to a concrete "
                "action sequence — not simply because the task is to reach a goal."
            ),
        ),
    ]


def get_ask_human_tool(
    wait_for_human_callback: Callable[[str], Coroutine[Any, Any, Optional[str]]],
):
    """Create an ask_human tool that delegates to the provided async callback.

    The callback should send the question to the human (e.g. via WebSocket),
    block until a response arrives, and return the response string.
    The human's answer is returned as a ToolMessage to the model.
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "langchain_core is required to build tools. Install langchain-core first."
        ) from exc

    async def _ask_human_async(question: str) -> str:
        print(f"[ask_human] Question to human: {question}")
        response = await wait_for_human_callback(question)
        if response and response.strip():
            print(f"[ask_human] Human responded: {response}")
            return response
        print("[ask_human] Human skipped without providing input.")
        return "(Human skipped without providing input.)"

    def _ask_human_sync(question: str) -> str:
        raise RuntimeError("ask_human requires an async execution context")

    return StructuredTool.from_function(
        func=_ask_human_sync,
        coroutine=_ask_human_async,
        name="ask_human",
        description=(
            "Ask your human collaborator a question and wait for their response. "
            "Use this when you need guidance, want to confirm a hypothesis, "
            "need clarification on ambiguous observations, or want human insight "
            "before committing to a direction. The human's answer is returned "
            "directly. Do not overuse this — make meaningful progress between asks."
        ),
    )


__all__ = ["get_hci_tools", "get_ask_human_tool"]
