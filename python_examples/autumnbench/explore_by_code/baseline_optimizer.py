"""Baseline optimizer: autonomous designer + executor loop.

This module contains the core async logic for the baseline workflow optimization
loop. It is called by the websocket handler but does not depend on it directly —
communication is done via a send_callback.
"""

import asyncio
import contextlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage

_FILE_DIR = Path(__file__).resolve().parent
_DESIGNER_DOCKERFILE = _FILE_DIR / "Dockerfile.designer"
_AUTUMNBENCH_DIR = _FILE_DIR.parent
_MARA_ROOT = _AUTUMNBENCH_DIR.parents[1]
for _p in [str(_FILE_DIR), str(_AUTUMNBENCH_DIR), str(_MARA_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_utils import get_llm  # noqa: E402
from env_wrapper import (  # noqa: E402
    create_pinned_runtime,
    get_env_tools_for_runtime,
    TEMPLATE_WORKSPACE_DIR,
    RUNS_WORKSPACE_ROOT_DIR,
)
from baseline_designer_prompt import (  # noqa: E402
    BASELINE_DESIGNER_SYSTEM_PROMPT,
    build_baseline_designer_prompt,
)
from planning_utils import (  # noqa: E402
    load_color_dict,
    goal_to_color_grid,
    color_grid_to_scene_graph,
    mask_to_positions,
)
from variant_env_utils import load_goal_and_mask  # noqa: E402

_EXAMPLE_BENCHMARK_DIR = (_AUTUMNBENCH_DIR / "example_benchmark").resolve()

logger = logging.getLogger(__name__)

SendCallback = Callable[[dict], Coroutine[Any, Any, None]]

_PARALLEL_VARIANT_CAPACITY = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_planning_goal(env_name: str):
    """Load planning goal/mask and convert to scene_graph format.

    Returns (goal_scene_graph_json, mask_scene_graph_json) as JSON strings,
    or (None, None) if no planning data exists.
    """
    data_dir, result = load_goal_and_mask(_EXAMPLE_BENCHMARK_DIR, env_name)
    if result is None:
        return None, None

    raw_goal, raw_mask = result
    color_dict = load_color_dict(data_dir)
    goal_color = goal_to_color_grid(raw_goal, color_dict)
    goal_sg = color_grid_to_scene_graph(goal_color)

    grid_size = len(goal_color)
    all_ones = all(raw_mask[r][c] == 1 for r in range(grid_size) for c in range(grid_size))
    if all_ones:
        mask_repr = "FULL_GRID"
    else:
        mask_repr = mask_to_positions(raw_mask)

    return (
        json.dumps(goal_sg, indent=2),
        json.dumps(mask_repr, indent=2) if not isinstance(mask_repr, str) else f'"{mask_repr}"',
    )


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)



# ---------------------------------------------------------------------------
# Designer phase
# ---------------------------------------------------------------------------

async def run_baseline_designer(
    *,
    designer_runtime,
    model: str,
    env_name: str,
    instruction: str,
    iteration: int,
    prev_workflow_code: Optional[str],
    send_callback: SendCallback,
    max_turns: int = 200,
) -> tuple[str, str | None]:
    """Run the designer LLM agent.

    Returns (workflow_code, mermaid_graph).
    mermaid_graph is None if compilation fails.
    """

    llm = get_llm(model=model)
    tools = get_env_tools_for_runtime(designer_runtime, timeout_seconds=60)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=BASELINE_DESIGNER_SYSTEM_PROMPT,
        middleware=[
            ModelCallLimitMiddleware(run_limit=max_turns, exit_behavior="end"),
        ],
    )

    user_prompt = build_baseline_designer_prompt(
        env_name=env_name,
        instruction=instruction,
        iteration=iteration,
        prev_workflow_code=prev_workflow_code,
    )

    final_messages = []
    async for event in agent.astream(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"recursion_limit": 1000},
        stream_mode="values",
    ):
        if isinstance(event, dict) and "messages" in event:
            final_messages = event.get("messages", []) or final_messages

    # Read workflow.py from the designer's workspace
    workflow_path = designer_runtime.active_workspace_dir / "workflow.py"
    if not workflow_path.is_file():
        raise FileNotFoundError(
            "Designer did not produce /workspace/workflow.py. "
            "The designer agent must write this file."
        )

    workflow_code = workflow_path.read_text(encoding="utf-8")

    # Compile to extract mermaid graph
    mermaid_graph = None
    try:
        _, mermaid_graph = _compile_workflow_and_mermaid(workflow_path, tools, llm)
    except Exception as exc:
        logger.warning(f"Failed to compile workflow for mermaid: {exc}")

    return workflow_code, mermaid_graph


# ---------------------------------------------------------------------------
# Import and compile the designer's workflow
# ---------------------------------------------------------------------------

def _import_workflow_module(workflow_path: Path):
    """Import workflow.py from the given path and return the module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_baseline_workflow", str(workflow_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workflow from {workflow_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compile_workflow_and_mermaid(workflow_path: Path, tools, llm):
    """Import workflow, compile graph, extract mermaid. Returns (compiled_graph, mermaid_str)."""
    mod = _import_workflow_module(workflow_path)
    if not hasattr(mod, "create_workflow"):
        raise AttributeError("workflow.py must define a create_workflow(tools, llm) function")
    compiled = mod.create_workflow(tools, llm)
    mermaid_str = compiled.get_graph().draw_mermaid()
    return compiled, mermaid_str


# ---------------------------------------------------------------------------
# Execution phase
# ---------------------------------------------------------------------------

async def run_baseline_execution(
    *,
    iteration: int,
    env_name: str,
    model: str,
    instruction: str,
    workflow_code: str,
    workflow_path: Path,
    workspace_template_dir: Path,
    variant_entries: List[Dict[str, Any]],
    run_dir: Path,
    send_callback: SendCallback,
) -> Dict[str, Dict[str, Any]]:
    """Execute the workflow on K variants in parallel. Returns per-variant results."""

    total_variants = len(variant_entries)
    semaphore = asyncio.Semaphore(_PARALLEL_VARIANT_CAPACITY)
    results: Dict[str, Dict[str, Any]] = {}
    results_lock = asyncio.Lock()

    iter_dir = run_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    async def run_variant(entry: Dict[str, Any]) -> None:
        variant_id = entry["id"]
        variant_workspace_dir = iter_dir / variant_id
        runtime = None

        async with semaphore:
            try:
                runtime = create_pinned_runtime(
                    env_name=variant_id,
                    task_mode="planning",
                    template_dir=workspace_template_dir,
                    workspace_dir=variant_workspace_dir,
                )
                runtime.install_batch_eval_env_client()
                runtime.set_client_control_mode("restricted")
                runtime.configure_environment(
                    env_name=variant_id,
                    task_mode="planning",
                )

                await send_callback({
                    "type": "baseline_execution_update",
                    "data": {
                        "iteration": iteration,
                        "variant_id": variant_id,
                        "status": "running",
                        "goal_reached": False,
                    },
                })

                # Import workflow and compile for this variant's tools/llm
                llm = get_llm(model=model)
                variant_tools = get_env_tools_for_runtime(runtime, timeout_seconds=30)
                compiled_graph, _ = _compile_workflow_and_mermaid(
                    workflow_path, variant_tools, llm
                )

                # Reset environment and get initial state
                initial_state = runtime.backend_reset()

                # Load goal and mask for this variant
                goal_sg, mask_sg = _load_planning_goal(variant_id)

                # Run the workflow
                node_logs = []
                full_node_states = []
                final_state = None
                async for event in compiled_graph.astream(
                    {
                        "initial_state": initial_state,
                        "env_name": variant_id,
                        "goal_scene_graph": goal_sg or "null",
                        "mask_scene_graph": mask_sg or "null",
                    },
                    stream_mode="updates",
                ):
                    # event is a dict of {node_name: node_output}
                    for node_name, node_output in event.items():
                        node_logs.append({
                            "node_name": node_name,
                            "status": "completed",
                            "output_summary": str(node_output)[:300] if node_output else None,
                        })
                        full_node_states.append({
                            "node_name": node_name,
                            "output": node_output,
                        })
                        final_state = node_output

                # Save trajectory and check goal
                trajectory_result = runtime.backend_save_trajectory(variant_id)
                goal_status = runtime.backend_goal_status()
                goal_reached = bool(goal_status.get("goal_reached"))

                trajectory_path = trajectory_result.get("saved_path")
                trajectory_payload = None
                if trajectory_path:
                    abs_path = Path(runtime.active_workspace_dir) / trajectory_path
                    if abs_path.is_file():
                        trajectory_payload = json.loads(
                            abs_path.read_text(encoding="utf-8")
                        )

                variant_result = {
                    "variant_id": variant_id,
                    "status": "succeeded",
                    "goal_reached": goal_reached,
                    "trajectory_path": trajectory_path,
                    "trajectory_payload": trajectory_payload,
                    "final_response": str(final_state)[:500] if final_state else "",
                    "error": None,
                    "node_logs": node_logs,
                    "full_node_states": full_node_states,
                }
            except Exception as exc:
                logger.error(
                    f"Baseline execution failed for {variant_id}: {exc}",
                    exc_info=True,
                )
                variant_result = {
                    "variant_id": variant_id,
                    "status": "failed",
                    "goal_reached": False,
                    "trajectory_path": None,
                    "trajectory_payload": None,
                    "final_response": "",
                    "error": str(exc),
                    "node_logs": [],
                    "full_node_states": [],
                }
            finally:
                if runtime is not None:
                    with contextlib.suppress(Exception):
                        runtime.set_client_control_mode("full")
                    with contextlib.suppress(Exception):
                        runtime.close()

            async with results_lock:
                results[variant_id] = variant_result

            # Exclude full_node_states from WebSocket to avoid large payloads
            ws_result = {k: v for k, v in variant_result.items() if k != "full_node_states"}
            await send_callback({
                "type": "baseline_execution_update",
                "data": {"iteration": iteration, **ws_result},
            })

    tasks = [asyncio.create_task(run_variant(entry)) for entry in variant_entries]
    await asyncio.gather(*tasks)
    return results


# ---------------------------------------------------------------------------
# Main optimizer loop
# ---------------------------------------------------------------------------

async def run_baseline_optimizer(
    *,
    env_name: str,
    model: str,
    instruction: str,
    variant_entries: List[Dict[str, Any]],
    max_iterations: int,
    send_callback: SendCallback,
    continue_event: asyncio.Event,
) -> None:
    """Main baseline optimizer loop.

    Args:
        send_callback: async callable that sends a dict as a JSON WebSocket message.
        continue_event: asyncio.Event that the caller sets to resume after pause.
    """

    run_id = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = RUNS_WORKSPACE_ROOT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Set up the designer's workspace
    designer_workspace = run_dir / "designer"
    shutil.copytree(TEMPLATE_WORKSPACE_DIR, designer_workspace)
    (designer_workspace / "workspace_template").mkdir(parents=True, exist_ok=True)
    # Copy the default template files into workspace_template so executors get them
    for item in TEMPLATE_WORKSPACE_DIR.iterdir():
        dest = designer_workspace / "workspace_template" / item.name
        if item.is_file() and not dest.exists():
            shutil.copy2(item, dest)
        elif item.is_dir() and not dest.exists():
            shutil.copytree(item, dest)
    (designer_workspace / "iterations").mkdir(parents=True, exist_ok=True)

    designer_runtime = None
    workflow_code: Optional[str] = None

    variant_ids = [e["id"] for e in variant_entries]

    try:
        # Create designer runtime with designer_workspace as its workspace.
        # Uses Dockerfile.designer which includes langgraph so the designer
        # can validate workflow.py inside the container.
        # No env_api_server — the designer only writes code and reads results.
        designer_runtime = create_pinned_runtime(
            task_mode="explore",
            workspace_dir=designer_workspace,
            dockerfile_path=str(_DESIGNER_DOCKERFILE),
            spawn_env_server=False,
        )

        await send_callback({
            "type": "baseline_optimizer_started",
            "data": {
                "run_id": run_id,
                "env_name": env_name,
                "variant_ids": variant_ids,
                "max_iterations": max_iterations,
            },
        })

        final_success_rate = 0.0

        for iteration in range(max_iterations):
            await send_callback({
                "type": "baseline_iteration_started",
                "data": {"iteration": iteration},
            })

            # --- Phase 1: Designer ---
            workflow_code, mermaid_graph = await run_baseline_designer(
                designer_runtime=designer_runtime,
                model=model,
                env_name=env_name,
                instruction=instruction,
                iteration=iteration,
                prev_workflow_code=workflow_code,
                send_callback=send_callback,
            )

            workflow_path = designer_workspace / "workflow.py"
            workspace_template_dir = designer_workspace / "workspace_template"

            await send_callback({
                "type": "baseline_designer_completed",
                "data": {
                    "iteration": iteration,
                    "workflow_code": workflow_code,
                    "mermaid_graph": mermaid_graph,
                },
            })

            # --- Phase 2: Execution ---
            await send_callback({
                "type": "baseline_execution_started",
                "data": {
                    "iteration": iteration,
                    "total_variants": len(variant_entries),
                },
            })

            results = await run_baseline_execution(
                iteration=iteration,
                env_name=env_name,
                model=model,
                instruction=instruction,
                workflow_code=workflow_code,
                workflow_path=workflow_path,
                workspace_template_dir=workspace_template_dir,
                variant_entries=variant_entries,
                run_dir=run_dir,
                send_callback=send_callback,
            )

            # Copy trajectories back to designer workspace for next iteration
            iter_dir = designer_workspace / "iterations" / f"{iteration:03d}"
            iter_traj_dir = iter_dir / "trajectories"
            iter_traj_dir.mkdir(parents=True, exist_ok=True)
            for vid, r in results.items():
                if r.get("trajectory_payload"):
                    traj_file = iter_traj_dir / f"{vid}.json"
                    traj_file.write_text(
                        json.dumps(r["trajectory_payload"], indent=2),
                        encoding="utf-8",
                    )

            # Write full execution logs to designer workspace
            iter_logs_dir = iter_dir / "execution_logs"
            iter_logs_dir.mkdir(parents=True, exist_ok=True)
            for vid, r in results.items():
                full_states = r.get("full_node_states", [])
                if full_states:
                    log_file = iter_logs_dir / f"{vid}.json"
                    try:
                        log_text = json.dumps(full_states, indent=2)
                    except (TypeError, ValueError):
                        log_text = json.dumps(full_states, indent=2, default=str)
                    log_file.write_text(log_text, encoding="utf-8")

            success_count = sum(1 for r in results.values() if r.get("goal_reached"))
            total = len(results)
            final_success_rate = success_count / total if total > 0 else 0.0

            await send_callback({
                "type": "baseline_iteration_completed",
                "data": {
                    "iteration": iteration,
                    "success_count": success_count,
                    "total_variants": total,
                    "success_rate": final_success_rate,
                },
            })

            # Check convergence
            if final_success_rate == 1.0:
                break

            # Pause between iterations
            if iteration < max_iterations - 1:
                continue_event.clear()
                await send_callback({
                    "type": "baseline_optimizer_paused",
                    "data": {"iteration": iteration},
                })
                await continue_event.wait()

        await send_callback({
            "type": "baseline_optimizer_completed",
            "data": {
                "run_id": run_id,
                "total_iterations": iteration + 1,
                "final_success_rate": final_success_rate,
            },
        })

    except asyncio.CancelledError:
        logger.info(f"Baseline optimizer {run_id} cancelled.")
    except Exception as exc:
        logger.error(f"Baseline optimizer failed: {exc}", exc_info=True)
        await send_callback({
            "type": "baseline_optimizer_error",
            "data": {"error": str(exc), "iteration": locals().get("iteration", -1)},
        })
    finally:
        if designer_runtime is not None:
            with contextlib.suppress(Exception):
                designer_runtime.close()
