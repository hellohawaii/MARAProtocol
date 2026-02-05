import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from pydantic import BaseModel, Field

from .langchain_utils import get_llm
from .prompts import SYSTEM_PROMPT_ORACLE_PROGRAM_CODE
from .python_wm_as_tool import PythonWorldModelTool

logger = logging.getLogger(__name__)


def _state_to_grid(state: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
    """Convert a scene graph state to a position -> color mapping."""
    grid: Dict[Tuple[int, int], str] = {}
    if not isinstance(state, dict):
        return grid
    for key, value in state.items():
        if key == "GRID_SIZE":
            continue
        if isinstance(value, list):
            for cell in value:
                if isinstance(cell, dict):
                    pos = cell.get("position")
                    color = cell.get("color", key)  # use key as color fallback
                    if isinstance(pos, dict) and "x" in pos and "y" in pos:
                        x, y = int(pos["x"]), int(pos["y"])
                        grid[(x, y)] = str(color)
    return grid


def _goal_to_grid(goal: Dict[str, Any]) -> Dict[Tuple[int, int], str]:
    """Convert a goal scene graph to a position -> color mapping."""
    grid: Dict[Tuple[int, int], str] = {}
    if not isinstance(goal, dict):
        return grid
    for key, value in goal.items():
        if key == "GRID_SIZE":
            continue
        color = str(key)
        if isinstance(value, list):
            for cell in value:
                if isinstance(cell, dict):
                    pos = cell.get("position", cell)
                    if isinstance(pos, dict) and "x" in pos and "y" in pos:
                        x, y = int(pos["x"]), int(pos["y"])
                        grid[(x, y)] = color
                    elif "x" in cell and "y" in cell:
                        x, y = int(cell["x"]), int(cell["y"])
                        grid[(x, y)] = color
    return grid


def _get_mask_positions(highlight_mask: Any, grid_size: int) -> Optional[set]:
    """Extract positions from highlight_mask. Returns None if FULL_GRID."""
    if highlight_mask == "FULL_GRID":
        return None
    if isinstance(highlight_mask, list):
        positions = set()
        for item in highlight_mask:
            if isinstance(item, dict) and "x" in item and "y" in item:
                positions.add((int(item["x"]), int(item["y"])))
        return positions
    return None


def _check_goal_reached(state: Any, goal: Any, highlight_mask: Any) -> bool:
    """Check if state satisfies goal within highlight_mask (programmatic, no LLM)."""
    if state is None or goal is None:
        return False
    
    state_grid = _state_to_grid(state)
    goal_grid = _goal_to_grid(goal)
    
    grid_size = state.get("GRID_SIZE", 16) if isinstance(state, dict) else 16
    mask_positions = _get_mask_positions(highlight_mask, grid_size)
    
    if mask_positions is None:
        positions_to_check = set(goal_grid.keys())
    else:
        positions_to_check = mask_positions & set(goal_grid.keys())
    
    if not positions_to_check:
        return True
    
    for pos in positions_to_check:
        if goal_grid.get(pos) != state_grid.get(pos):
            return False
    return True


class LATSInput(TypedDict):
    available_actions: List[str]
    goal: Any
    highlight_mask: Any


class TreeState(TypedDict):
    root: "Node"
    input: LATSInput


class RemainingStepsEstimate(BaseModel):
    """LLM output for estimating remaining steps (found_solution is determined programmatically)."""
    reasoning: str = Field(description="Reasoning for the estimation")
    remaining_steps: int = Field(description="Estimated number of steps to reach the goal. Use 100 or -1 if impossible.")


class Reflection(BaseModel):
    reasoning: str = Field(description="Reasoning for the evaluation")
    remaining_steps: int = Field(description="Estimated number of steps to reach the goal. Use 100 or -1 if impossible.")
    found_solution: bool = Field(description="Whether the goal is fully satisfied")

    @property
    def normalized_score(self) -> float:
        if self.found_solution:
            return 0.0
        if self.remaining_steps is None or self.remaining_steps < 0:
            return -100.0
        return -float(min(self.remaining_steps, 100))


class Node:
    def __init__(
        self,
        state: Any,
        hidden_state: Any,
        reflection: Reflection,
        node_id: int,
        parent: Optional["Node"] = None,
        action: Optional[str] = None,
    ) -> None:
        self.state = state
        self.hidden_state = hidden_state
        self.reflection = reflection
        self.id = node_id
        self.parent = parent
        self.action = action
        self.children: List["Node"] = []
        self.value = reflection.normalized_score
        self.visits = 1
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution
        if self._is_solved:
            self._mark_tree_as_solved()

    @property
    def is_solved(self) -> bool:
        return self._is_solved

    @property
    def is_terminal(self) -> bool:
        return not self.children

    def backpropagate(self, reward: float) -> None:
        node: Optional[Node] = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def upper_confidence_bound(self, exploration_weight: float = 1.0) -> float:
        if self.parent is None:
            return self.value
        if self.visits == 0:
            return self.value
        average_reward = self.value
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def get_trajectory_actions(self) -> List[str]:
        actions: List[str] = []
        node: Optional[Node] = self
        while node:
            if node.action is not None:
                actions.append(node.action)
            node = node.parent
        return actions[::-1]

    def _mark_tree_as_solved(self) -> None:
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent

    def _get_all_children(self) -> List["Node"]:
        nodes = [self]
        all_nodes: List[Node] = []
        while nodes:
            current = nodes.pop()
            all_nodes.extend(current.children)
            nodes.extend(current.children)
        return all_nodes

    def get_best_solution(self) -> "Node":
        all_nodes = [self] + self._get_all_children()
        return max(
            all_nodes,
            key=lambda node: (int(node.is_solved), node.value, -node.depth),
        )


class LATSPlanner:
    def __init__(
        self,
        env_name: str,
        use_obfuscated_code: bool = False,
        data_dir: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: str = "openai/gpt-4o",
        max_depth: int = 6,
        max_rollouts: int = 20,
        n_candidates: int = 5,
        exploration_weight: float = 1.2,
        logging_path: Optional[str] = None,
    ) -> None:
        self.env_name = env_name
        self.use_obfuscated_code = use_obfuscated_code
        self.data_dir = data_dir
        self.llm = get_llm(model=llm_model)
        self.max_depth = max_depth
        self.max_rollouts = max_rollouts
        self.n_candidates = n_candidates
        self.exploration_weight = exploration_weight
        self.logging_path = logging_path
        self._log: Optional[Dict[str, Any]] = None
        self._log_file: Optional[str] = None
        self._node_counter = 0

        self.wm_tool = PythonWorldModelTool(
            env_name=env_name,
            use_obfuscated_code=use_obfuscated_code,
            data_dir=data_dir,
        )
        self.wm_tool._ensure_loaded()

    def plan(self, observation_text: str, available_actions: List[str]) -> Optional[List[str]]:
        payload = _extract_json_payload(observation_text)

        
        goal = payload.get("goal")
        highlight_mask = payload.get("highlight_mask")
        render = payload.get("render")
        if goal is None or highlight_mask is None or render is None:
            raise RuntimeError ("LATSPlanner: missing render, goal or highlight_mask in observation.")

        init_state = self.wm_tool._state
        init_hidden = self.wm_tool._hidden_state
        if init_state is None:
            raise RuntimeError("LATSPlanner: initial world model state unavailable.")

        self._start_log(observation_text, payload, available_actions)

        reflection = self._evaluate_state(init_state, goal, highlight_mask)
        root = Node(
            state=init_state,
            hidden_state=init_hidden,
            reflection=reflection,
            node_id=self._next_node_id(),
        )
        self._log_event(
            "root_created",
            node=self._node_snapshot(root),
        )

        graph = self._build_graph()
        input_state: TreeState = {
            "root": root,
            "input": {
                "available_actions": available_actions,
                "goal": goal,
                "highlight_mask": highlight_mask,
            },
        }

        last_state: Optional[TreeState] = None
        recursion_limit = max(25, int(self.max_rollouts))
        for step in graph.stream(input_state, {"recursion_limit": recursion_limit}):
            _, last_state = next(iter(step.items()))
            if last_state["root"].is_solved:
                break

        if last_state is None:
            self._finalize_log(None, None)
            return None

        best_node = last_state["root"].get_best_solution()
        actions = best_node.get_trajectory_actions()
        self._log_event(
            "best_solution",
            node=self._node_snapshot(best_node),
            actions=actions,
        )
        self._finalize_log(best_node, actions)
        return actions or None

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(TreeState)
        builder.add_node("start", self._generate_initial_response)
        builder.add_node("expand", self._expand)
        builder.add_edge(START, "start")
        builder.add_conditional_edges("start", self._should_loop, ["expand", END])
        builder.add_conditional_edges("expand", self._should_loop, ["expand", END])
        return builder.compile()

    def _should_loop(self, state: TreeState) -> str:
        root = state["root"]
        if root.is_solved:
            return END
        if root.visits >= self.max_rollouts:
            return END
        return "expand"

    def _generate_initial_response(self, state: TreeState) -> TreeState:
        return state

    def _expand(self, state: TreeState) -> TreeState:
        root = state["root"]
        best_candidate = self._select(root)
        partial_plan = best_candidate.get_trajectory_actions()
        remaining_steps = best_candidate.reflection.remaining_steps
        print(
            f"[LATS expand] depth={best_candidate.depth} remaining_steps={remaining_steps} "
            f"partial_plan={partial_plan}"
        )
        self._log_event(
            "select",
            node=self._node_snapshot(best_candidate),
        )
        
        # If the node cannot be expanded (max depth reached or impossible state),
        # we treat it as a terminal leaf effectively visited again.
        # We backpropagate to update visits/value so UCT can select other nodes.
        if (
            best_candidate.depth >= self.max_depth
            or best_candidate.reflection.remaining_steps == -1
        ):
            self._log_event(
                "terminal_leaf",
                node=self._node_snapshot(best_candidate),
                reason="max_depth" if best_candidate.depth >= self.max_depth else "impossible",
            )
            best_candidate.backpropagate(best_candidate.value)
            return state

        available_actions = state["input"]["available_actions"]
        goal = state["input"]["goal"]
        highlight_mask = state["input"]["highlight_mask"]

        candidates = self._generate_actions(
            best_candidate.state,
            goal,
            highlight_mask,
            available_actions,
        )
        self._log_event(
            "candidates",
            node_id=best_candidate.id,
            actions=candidates,
        )

        if not candidates:
            # No actions generated (e.g. LLM failure or empty), treat as visited terminal
            self._log_event(
                "no_candidates",
                node=self._node_snapshot(best_candidate),
            )
            best_candidate.backpropagate(best_candidate.value)
            return state

        pending_indices: List[int] = []
        pending_states: List[Any] = []
        child_specs: List[Tuple[Any, Any, str, Optional[Reflection]]] = []

        for action in candidates:
            next_state, next_hidden = self._simulate(best_candidate, action)
            if next_state is None:
                self._log_event(
                    "simulate_failed",
                    parent_id=best_candidate.id,
                    action=action,
                )
                continue
            # Handle quit and solved states before LLM evaluation
            if next_state == "__QUIT__":
                reflection = Reflection(
                    reasoning="Quit action executed. Goal not reached, cannot continue.",
                    remaining_steps=-1,
                    found_solution=False,
                )
                next_state = best_candidate.state
                next_hidden = best_candidate.hidden_state
                child_specs.append((next_state, next_hidden, action, reflection))
                self._log_event(
                    "simulate_quit",
                    parent_id=best_candidate.id,
                    action=action,
                )
                continue
            if _check_goal_reached(next_state, goal, highlight_mask):
                reflection = Reflection(
                    reasoning="Goal reached: all highlighted positions match the goal state.",
                    remaining_steps=0,
                    found_solution=True,
                )
                child_specs.append((next_state, next_hidden, action, reflection))
                self._log_event(
                    "simulate_goal_reached",
                    parent_id=best_candidate.id,
                    action=action,
                )
                continue

            pending_indices.append(len(child_specs))
            pending_states.append(next_state)
            child_specs.append((next_state, next_hidden, action, None))

        if pending_states:
            reflections = self._evaluate_state_batch(pending_states, goal, highlight_mask)
            for idx, reflection in zip(pending_indices, reflections):
                state_i, hidden_i, action_i, _ = child_specs[idx]
                child_specs[idx] = (state_i, hidden_i, action_i, reflection)
            self._log_event(
                "evaluated_batch",
                parent_id=best_candidate.id,
                reflections=[self._reflection_snapshot(r) for r in reflections],
            )

        for next_state, next_hidden, action, reflection in child_specs:
            if reflection is None:
                reflection = Reflection(
                    reasoning="LLM reflection missing.",
                    remaining_steps=-1,
                    found_solution=False,
                )
            child = Node(
                state=next_state,
                hidden_state=next_hidden,
                reflection=reflection,
                node_id=self._next_node_id(),
                parent=best_candidate,
                action=action,
            )
            child.backpropagate(reflection.normalized_score)
            best_candidate.children.append(child)
            self._log_event(
                "child_added",
                parent_id=best_candidate.id,
                child=self._node_snapshot(child),
                action=action,
                reflection=self._reflection_snapshot(reflection),
            )
            self._log_event(
                "backpropagate",
                node_id=child.id,
                reward=reflection.normalized_score,
            )

        return state

    def _select(self, root: Node) -> Node:
        node = root
        while node.children:
            node = max(
                node.children,
                key=lambda child: child.upper_confidence_bound(self.exploration_weight),
            )
        return node

    def _simulate(self, node: Node, action: str) -> Tuple[Any, Any]:
        # Handle quit action specially - python environment doesn't support it
        if action.strip().lower() == "quit":
            return "__QUIT__", None
        
        action_payload = _parse_action(action)
        if action_payload is None:
            return None, None
        predict_fn = self.wm_tool._predict_dynamics_func
        if predict_fn is None:
            return None, None
        next_state, next_hidden_state, error = _call_predict_safely(
            predict_fn,
            node.state,
            node.hidden_state,
            action_payload,
        )
        if error:
            logger.debug(f"LATSPlanner simulate error: {error}")
            return None, None
        return next_state, next_hidden_state

    def _generate_actions(
        self,
        state: Any,
        goal: Any,
        highlight_mask: Any,
        available_actions: List[str],
    ) -> List[str]:
        prompt = SYSTEM_PROMPT_ORACLE_PROGRAM_CODE
        if self.wm_tool._code_str:
            prompt = (
                f"{prompt}\n\n"
                f"You have found that the code of ground truth world model is:\n"
                f"{self.wm_tool._code_str}"
            )

        prompt += (
            "\n\nGiven the current state, goal, and highlight mask, propose the next action to try. "
            "Actions must be from the available actions or valid click coordinates."
        )
        content = json.dumps(
            {
                "state": state,
                "goal": goal,
                "highlight_mask": highlight_mask,
                "available_actions": available_actions,
            },
            ensure_ascii=False,
        )

        try:
            # Use the provided world model tool.
            tools = self.wm_tool.get_langgraph_tools()
            llm_with_tools = self.llm.bind_tools(tools, tool_choice="any")
            
            # Generate N candidates by batching requests with explicit seeds in messages
            messages_batch = []
            for i in range(self.n_candidates):
                messages_batch.append([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content, "seed": i}
                ])
            
            responses = llm_with_tools.batch(messages_batch)
            
            actions = []
            for ai_msg in responses:
                self._log_event(
                    "llm_generate_actions_response",
                    response=self._safe_serialize(ai_msg),
                    tool_calls=self._safe_serialize(getattr(ai_msg, "tool_calls", None)),
                )
                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == "python_wm_step":
                        action_arg = tool_call["args"].get("action")
                        if action_arg:
                            actions.append(str(action_arg))
                            # Only take one action per generation
                            break
            
        except Exception as e:
            logger.warning(f"LLM generate actions failed: {e}")
            actions = []

        filtered: List[str] = []
        seen = set()
        for action in actions:
            if isinstance(action, str) and _is_action_allowed(action, available_actions):
                if action not in seen:
                    filtered.append(action)
                    seen.add(action)
            else:
                logger.debug(f"LATSPlanner: filtered out invalid action '{action}'")
        if filtered:
            return filtered
        else:
            logger.debug("LATSPlanner: no valid actions generated by LLM, falling back to sampling.")
        fallback = _sample_actions(available_actions, self.n_candidates)
        return fallback

    def _evaluate_state(self, state: Any, goal: Any, highlight_mask: Any) -> Reflection:
        """Evaluate state with LLM for remaining steps (goal check handled in expand)."""
        if state is None or goal is None:
            raise RuntimeError("LATSPlanner: state or goal is None in evaluation.")
        
        # Goal not reached: use LLM to estimate remaining steps
        prompt = (
            "You are a helpful assistant operating with access to the ground-truth world model for a interactive grid environment.\n"
            "The goal has NOT been reached yet. Given the current state, goal state, and highlight mask, "
            "estimate the remaining steps to reach the goal. "
            "Only the positions indicated by highlight_mask matter for success; all other positions are ignored. "
            "If the goal is unreachable, return -1 for remaining steps.\n\n"
            "Here is a description of the actions:"
            "- `click x y` - Click on the cell at the location (x, y) on the grid. "
            "For a grid of size GRID_SIZE, x and y must be between 0 and GRID_SIZE-1 inclusive.\n"
            "- `left` - Press the left arrow key.\n"
            "- `right` - Press the right arrow key.\n"
            "- `up` - Press the up arrow key.\n"
            "- `down` - Press the down arrow key.\n"
            "- `noop` - Do nothing and continue to the next step.\n\n"
        )

        if self.wm_tool._code_str:
            prompt += (
                f"Here is the code of the ground truth world model:\n"
                f"{self.wm_tool._code_str}"
            )

        content = json.dumps(
            {
                "state": state,
                "goal": goal,
                "highlight_mask": highlight_mask,
            },
            ensure_ascii=False,
        )

        try:
            evaluator = self.llm.with_structured_output(RemainingStepsEstimate)
            estimate = evaluator.invoke([
                ("system", prompt),
                ("user", content)
            ])
            self._log_event(
                "llm_reflection_response",
                response=self._safe_serialize(estimate),
            )
            result = Reflection(
                reasoning=estimate.reasoning,
                remaining_steps=estimate.remaining_steps,
                found_solution=False,
            )
        except Exception as e:
            logger.warning(f"LLM reflection failed: {e}")
            result = Reflection(
                reasoning="LLM reflection failed.",
                remaining_steps=-1,
                found_solution=False,
            )

        return result

    def _evaluate_state_batch(
        self,
        states: List[Any],
        goal: Any,
        highlight_mask: Any,
    ) -> List[Reflection]:
        """Batch version of _evaluate_state for parallel LLM calls."""
        if not states:
            return []
        if goal is None:
            return [
                Reflection(
                    reasoning="Missing goal.",
                    remaining_steps=-1,
                    found_solution=False,
                )
                for _ in states
            ]

        prompt = (
            "You are a helpful assistant operating with access to the ground-truth world model for a interactive grid environment.\n"
            "The goal has NOT been reached yet. Given the current state, goal state, and highlight mask, "
            "estimate the remaining steps to reach the goal. "
            "Only the positions indicated by highlight_mask matter for success; all other positions are ignored. "
            "If the goal is unreachable, return -1 for remaining steps.\n\n"
            "Here is a description of the actions:"
            "- `click x y` - Click on the cell at the location (x, y) on the grid. "
            "For a grid of size GRID_SIZE, x and y must be between 0 and GRID_SIZE-1 inclusive.\n"
            "- `left` - Press the left arrow key.\n"
            "- `right` - Press the right arrow key.\n"
            "- `up` - Press the up arrow key.\n"
            "- `down` - Press the down arrow key.\n"
            "- `noop` - Do nothing and continue to the next step.\n\n"
        )

        if self.wm_tool._code_str:
            prompt += (
                f"Here is the code of the ground truth world model:\n"
                f"{self.wm_tool._code_str}"
            )

        messages_batch = []
        for state in states:
            content = json.dumps(
                {
                    "state": state,
                    "goal": goal,
                    "highlight_mask": highlight_mask,
                },
                ensure_ascii=False,
            )
            messages_batch.append([
                ("system", prompt),
                ("user", content),
            ])

        try:
            evaluator = self.llm.with_structured_output(RemainingStepsEstimate)
            estimates = evaluator.batch(messages_batch)
            results: List[Reflection] = []
            for estimate in estimates:
                self._log_event(
                    "llm_reflection_response",
                    response=self._safe_serialize(estimate),
                )
                results.append(
                    Reflection(
                        reasoning=estimate.reasoning,
                        remaining_steps=estimate.remaining_steps,
                        found_solution=False,
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"LLM reflection batch failed: {e}")
            return [
                Reflection(
                    reasoning="LLM reflection failed.",
                    remaining_steps=-1,
                    found_solution=False,
                )
                for _ in states
            ]

    def _next_node_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def _start_log(
        self,
        observation_text: str,
        payload: Dict[str, Any],
        available_actions: List[str],
    ) -> None:
        if not self.logging_path:
            return
        log_dir = os.path.join(self.logging_path, self.env_name, "lats")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        self._log_file = os.path.join(log_dir, f"lats_tree_search_{timestamp}.json")
        self._log = {
            "env_name": self.env_name,
            "max_depth": self.max_depth,
            "max_rollouts": self.max_rollouts,
            "n_candidates": self.n_candidates,
            "exploration_weight": self.exploration_weight,
            "timestamp_ms": timestamp,
            "observation_text": observation_text,
            "available_actions": available_actions,
            "payload": payload,
            "events": [],
        }

    def _log_event(self, event_type: str, **data: Any) -> None:
        if not self._log:
            return
        event = {
            "type": event_type,
            "ts_ms": int(time.time() * 1000),
        }
        event.update(data)
        self._log["events"].append(event)

    def _finalize_log(self, best_node: Optional[Node], actions: Optional[List[str]]) -> None:
        if not self._log or not self._log_file:
            return
        if best_node is not None:
            self._log["best_node"] = self._node_snapshot(best_node)
            self._log["best_actions"] = actions or []
        try:
            with open(self._log_file, "w") as f:
                json.dump(self._log, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"Failed to write LATS log: {exc}")

    def _node_snapshot(self, node: Node) -> Dict[str, Any]:
        return {
            "id": node.id,
            "depth": node.depth,
            "visits": node.visits,
            "value": node.value,
            "action": node.action,
            "is_solved": node.is_solved,
            "remaining_steps": node.reflection.remaining_steps,
            "found_solution": node.reflection.found_solution,
        }

    def _reflection_snapshot(self, reflection: Reflection) -> Dict[str, Any]:
        return {
            "reasoning": reflection.reasoning,
            "remaining_steps": reflection.remaining_steps,
            "found_solution": reflection.found_solution,
        }

    def _safe_serialize(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {k: self._safe_serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._safe_serialize(v) for v in value]
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                return str(value)
        if hasattr(value, "dict"):
            try:
                return value.dict()
            except Exception:
                return str(value)
        return str(value)


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    blob = _extract_json_blob(text)
    if not blob:
        return None
    try:
        payload = json.loads(blob)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return None
    return None


def _extract_json_blob(text: str) -> Optional[str]:
    if not text:
        return None
    
    # Try finding the start of JSON payload by looking for known keys
    # "render", "goal", "highlight_mask" are keys we expect.
    # The JSON structure starts with {"render": ...}
    
    # Heuristic: Find first occurrence of {"render": 
    start_anchor = '{"render":'
    start_pos = text.find(start_anchor)
    
    if start_pos == -1:
        # Fallback to generic structure if specific key not keys
        candidates_starts = []
    else:
        candidates_starts = [start_pos]
        
    # Also find all { if fallback needed
    if not candidates_starts:
        pos = 0
        while True:
            pos = text.find("{", pos)
            if pos == -1:
                break
            candidates_starts.append(pos)
            pos += 1
            
        pos = 0
        while True:
            pos = text.find("[", pos)
            if pos == -1:
                break
            candidates_starts.append(pos)
            pos += 1
        
        candidates_starts.sort()
    
    end_brace = text.rfind("}")
    end_bracket = text.rfind("]")
    
    if end_brace == -1 and end_bracket == -1:
        return None
        
    for start in candidates_starts:
        # Determine likely end
        if text[start] == '{':
            end = end_brace
        else:
            end = end_bracket
            
        if end <= start:
            continue
            
        candidate = text[start : end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
            
    return None


def _parse_action(action: str) -> Optional[Dict[str, Any]]:
    action = action.strip().lower()
    if action in {"left", "right", "up", "down", "noop"}:
        return {"type": action}
    click_match = re.match(r"click\s+(\d+)\s+(\d+)", action)
    if click_match:
        x, y = map(int, click_match.groups())
        return {"type": "click", "x": x, "y": y}
    return None


def _is_action_allowed(action: str, available_actions: List[str]) -> bool:
    if action in available_actions:
        return True
    click_match = re.match(r"click\s+(\d+)\s+(\d+)", action)
    if click_match:
        for candidate in available_actions:
            if candidate.startswith("click ["):
                return True
    return False


def _sample_actions(available_actions: List[str], n: int) -> List[str]:
    if not available_actions:
        return []
    if len(available_actions) <= n:
        return available_actions
    return random.sample(available_actions, n)


def _call_predict_safely(
    predict_fn: Any,
    state: Any,
    hidden_state: Any,
    action: Any,
) -> Tuple[Any, Any, Optional[str]]:
    try:
        result = predict_fn(state, hidden_state, action)
    except Exception as exc:
        return None, None, str(exc)
    if not isinstance(result, tuple) or len(result) != 2:
        return None, None, "predict_dynamics() must return (next_state, next_hidden_state)."
    return result[0], result[1], None
