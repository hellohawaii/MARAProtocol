"""
Exploration subgraph — question generation, ReAct agent, and structured
summarization.

This module implements a LangGraph subgraph that:
1. Generates exploration questions (if none were provided by the refine stage).
2. Runs a ReAct agent to interactively explore an Autumn environment, collecting
   trajectories via step/reset/stop tools.
3. Summarises the exploration session by answering the investigation questions
   using ``llm.with_structured_output``.

The subgraph handles both "free" and "targeted" exploration: when
``pending_questions`` is pre-populated (from the refine subgraph), it skips
question generation; otherwise it generates its own questions first.  A
``questions_from_refine`` flag in the output state indicates which path was
taken.

The subgraph is symmetric with ``code_refine_graph.py``: the main orchestrator
in ``baseline_explorer.py`` invokes both subgraphs in alternation.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup (mirrors baseline_explorer.py — needed for interpreter_module)
# ---------------------------------------------------------------------------
_FILE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FILE_DIR.parents[2]  # /app
for _p in [str(_REPO_ROOT), str(_FILE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interpreter_module import Interpreter  # noqa: E402
from autumnstdlib import autumnstdlib  # noqa: E402

logger = logging.getLogger(__name__)


# ============================================================
# Pydantic Models for Structured Summarisation
# ============================================================

class ExplorationQuestion(BaseModel):
    """A single scientific question to investigate during exploration."""
    question: str = Field(
        description="A specific, actionable scientific question about the "
                    "environment dynamics to investigate"
    )


class ExplorationQuestions(BaseModel):
    """Structured output for the question-generation node."""
    exploration_sufficient: bool = Field(
        default=False,
        description="Set to True if the current knowledge and code are "
                    "sufficient to fully describe the environment dynamics "
                    "and no further exploration is needed.  When True the "
                    "questions list may be empty.",
    )
    questions: List[ExplorationQuestion] = Field(
        description="2-3 scientific questions to investigate in the next "
                    "exploration round.  May be empty when "
                    "exploration_sufficient is True."
    )


class ExplorationQA(BaseModel):
    """A single scientific question and its answer from exploration."""
    question: str = Field(
        description="A scientific question about the environment dynamics"
    )
    answer: str = Field(
        description="The answer or finding based on the exploration"
    )


class ExplorationSummary(BaseModel):
    """Structured output produced by the summarise node."""
    qa_pairs: List[ExplorationQA] = Field(
        description="List of scientific questions investigated and their answers"
    )


# ============================================================
# Self-contained Exploration Environment
# ============================================================

class ExplorationEnvironment:
    """Lightweight environment wrapper around the Autumn Interpreter.

    - Accepts a direct path to the ``.sexp`` program (no ``tests/`` assumption)
    - Has no gRPC, TypeMapper, or matplotlib dependencies
    - Records trajectories in the executor format expected by
      ``code_refine_graph.execute_code_on_trajectories``
    - Optionally applies object-type obfuscation via ``obfuscation_mapping.json``
    """

    def __init__(
        self,
        program_path: str,
        *,
        seed: int = 0,
        max_steps: int = 200,
        use_obfuscation: bool = False,
        obfuscation_mapping_path: Optional[str] = None,
    ):
        with open(program_path, "r") as f:
            self.prog = f.read()
        self.seed = seed
        self.max_steps = max_steps
        self.time = 0
        self.interpreter: Optional[Interpreter] = None
        self._trajectory: List[Dict[str, Any]] = []

        # --- obfuscation ---
        self._obfuscation_map: Optional[Dict[str, str]] = None
        if use_obfuscation:
            if obfuscation_mapping_path is None:
                candidates = [
                    os.path.join(os.path.dirname(program_path), "obfuscation_mapping.json"),
                    os.path.join(os.path.dirname(program_path), "..", "obfuscation_mapping.json"),
                    os.path.join(_FILE_DIR, "example_benchmark", "obfuscation_mapping.json"),
                ]
                for c in candidates:
                    if os.path.isfile(c):
                        obfuscation_mapping_path = c
                        break
            if obfuscation_mapping_path and os.path.isfile(obfuscation_mapping_path):
                with open(obfuscation_mapping_path, "r") as f:
                    self._obfuscation_map = json.load(f)
                logger.info("Obfuscation enabled with %d mappings from %s",
                            len(self._obfuscation_map), obfuscation_mapping_path)
            else:
                logger.warning("use_obfuscation=True but no mapping file found; "
                               "obfuscation disabled.")

    # ---- helpers ----

    def _get_raw_frame(self) -> Dict[str, Any]:
        raw = json.loads(self.interpreter.render_all())
        if self._obfuscation_map:
            raw = self._obfuscate_state(raw)
        return raw

    def _obfuscate_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        obfuscated: Dict[str, Any] = {}
        for key, value in state.items():
            new_key = self._obfuscation_map.get(key, key)
            obfuscated[new_key] = value
        return obfuscated

    @staticmethod
    def _parse_action(action_str: str) -> Optional[Dict[str, Any]]:
        action_str = action_str.strip().lower()
        if action_str in ("left", "right", "up", "down", "noop"):
            return {"type": action_str}
        m = re.match(r"click\s+(\d+)\s+(\d+)", action_str)
        if m:
            return {"type": "click", "x": int(m.group(1)), "y": int(m.group(2))}
        return None

    def _apply_action(self, action_str: str) -> bool:
        first_line = action_str.strip().split("\n")[0].strip()
        if first_line.startswith("click"):
            parts = first_line.split()
            if len(parts) >= 3:
                self.interpreter.click(int(parts[1]), int(parts[2]))
                return True
            return False
        mapping = {"left": self.interpreter.left, "right": self.interpreter.right,
                   "up": self.interpreter.up, "down": self.interpreter.down,
                   "noop": lambda: None}
        fn = mapping.get(first_line)
        if fn:
            fn()
            return True
        return False

    # ---- public API ----

    def reset(self):
        self.interpreter = Interpreter()
        self.interpreter.run_script(self.prog, autumnstdlib, "", self.seed)
        self.time = 0
        self._trajectory = [{"action": None, "rawFrame": self._get_raw_frame()}]

    def step(self, action_str: str) -> str:
        if self.time >= self.max_steps:
            return f"Max steps ({self.max_steps}) reached."
        self.time += 1
        parsed = self._parse_action(action_str)
        if not self._apply_action(action_str):
            return f"Invalid action: {action_str}"
        self.interpreter.step()
        raw = self._get_raw_frame()
        if parsed:
            self._trajectory.append({"action": parsed, "rawFrame": raw})
        return json.dumps(raw, ensure_ascii=False)

    def get_observation(self) -> str:
        return json.dumps(self._get_raw_frame(), ensure_ascii=False)

    def get_raw_trajectory(self) -> List[Dict[str, Any]]:
        return list(self._trajectory)

    def get_action_space_description(self) -> str:
        raw = self._get_raw_frame()
        gs = raw.get("GRID_SIZE", 0)
        return (
            f"click x y (where 0 <= x,y <= {gs - 1}), "
            "left, right, up, down, noop"
        )


# ============================================================
# Trajectory Conversion
# ============================================================

def convert_env_trajectory(env_traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert environment trajectory list to executor format.

    Environment stores::

        [
            {"action": None, "rawFrame": S0},  # initial
            {"action": A0,   "rawFrame": S1},  # after A0
            ...
        ]

    The code executor expects::

        {
            "frames": [{"rawFrame": S0}, {"rawFrame": S1}, ...],
            "frameActions": [A0, A1, ...]
        }
    """
    frames = [{"rawFrame": step["rawFrame"]} for step in env_traj]
    actions = [step["action"] for step in env_traj if step["action"] is not None]
    return {"frames": frames, "frameActions": actions}


# ============================================================
# Exploration System Prompt Builder
# ============================================================

_EXPLORE_SYSTEM_PROMPT = """\
You are a scientist exploring an interactive grid environment to discover its \
underlying dynamics (transition function).  Your goal is to collect diverse, \
informative trajectories so that a code-generation system can write a \
`predict_dynamics(visible_state, hidden_state, action)` function.

**Available tools:**
- `env_step(action)` — take an action.  Valid actions: \
`click x y`, `left`, `right`, `up`, `down`, `noop`.
- `env_reset()` — reset the environment and start collecting a new trajectory.  \
Use this to gather multiple independent trajectories.
- `stop_exploration()` — call this when you have completed your current \
exploration plan.

**When to stop:**
You do NOT need to collect exhaustively complete data before stopping.  This is \
an iterative process — you will be called again later to explore further.  Stop when:
- You have collected a few trajectories that cover the basic dynamics (enough for \
a reasonable first draft of the code), OR
- You have answered the investigation questions listed below, OR
- Your current exploration plan is finished.

**Strategy:**
- Try different actions systematically to understand cause-and-effect.
- Use `env_reset()` to collect multiple trajectories with different action sequences.
- Pay attention to objects, their colors, positions, and how they change.
- Consider whether hidden state (e.g. velocities, counters, modes) exists.
- If investigation questions are listed below, focus on answering them first.
- Call `stop_exploration()` once your plan is done.  Do not over-collect.
"""

_EXPLORE_CODE_SECTION = """
**Current predict_dynamics code (for reference):**
```python
{code}
```
"""

_EXPLORE_QA_SECTION = """
**Scientific knowledge gathered so far (Q&A):**
{qa_text}
"""

_EXPLORE_PENDING_SECTION = """
**IMPORTANT — Investigation questions from the code-generation system:**
The code-generation system tried to write `predict_dynamics` code but encountered \
errors it could not resolve from existing data alone.  The following questions \
describe specific aspects of the dynamics that are still uncertain.  Please design \
your exploration to collect trajectories that answer these questions.

{questions}
"""

_EXPLORE_ERRORS_SECTION = """
**Frames where the current code makes wrong predictions (for context):**
{errors}
"""


def _format_qa_pairs(qa_list: List[Dict[str, str]]) -> str:
    """Format a list of Q&A dicts into readable text."""
    if not qa_list:
        return "(none yet)"
    parts: List[str] = []
    for i, qa in enumerate(qa_list, 1):
        parts.append(f"Q{i}: {qa.get('question', '?')}")
        parts.append(f"A{i}: {qa.get('answer', '?')}")
        parts.append("")
    return "\n".join(parts)


def build_exploration_prompt(
    current_code: Optional[str],
    explored_qa: List[Dict[str, str]],
    pending_questions: List[str],
    error_frames: List[Dict[str, Any]],
) -> str:
    """Build the system prompt for the exploration agent."""
    prompt = _EXPLORE_SYSTEM_PROMPT

    if current_code:
        prompt += _EXPLORE_CODE_SECTION.format(code=current_code)

    if explored_qa:
        prompt += _EXPLORE_QA_SECTION.format(qa_text=_format_qa_pairs(explored_qa))

    if pending_questions:
        qtext = "\n".join(f"- {q}" for q in pending_questions)
        prompt += _EXPLORE_PENDING_SECTION.format(questions=qtext)

    if error_frames:
        from code_refine_graph import format_error_frames_text
        prompt += _EXPLORE_ERRORS_SECTION.format(
            errors=format_error_frames_text(error_frames))

    return prompt


# ============================================================
# Exploration Tools Factory
# ============================================================

def create_exploration_tools(
    env: ExplorationEnvironment,
    collected_trajectories: List[Dict[str, Any]],
):
    """Create LangChain tools for the exploration agent.

    ``collected_trajectories`` is a *mutable* list; env_reset and stop_exploration
    will append converted trajectories to it.

    Returns (tools_list, stop_flag) where stop_flag is a dict
    ``{"stopped": bool}`` mutated by stop_exploration.
    """
    stop_flag = {"stopped": False}

    @tool(description=(
        "Take an action in the environment. "
        "Valid actions: 'click x y', 'left', 'right', 'up', 'down', 'noop'. "
        "Returns the observation after the action."
    ))
    def env_step(action: str) -> str:
        """Take an action in the environment and return the new observation."""
        try:
            return env.step(action)
        except Exception as e:
            return f"Error: {e}"

    @tool(description=(
        "Reset the environment to its initial state. "
        "This saves the current trajectory and starts a new one. "
        "Returns the initial observation of the fresh environment."
    ))
    def env_reset() -> str:
        """Reset the environment, saving the current trajectory."""
        raw_traj = env.get_raw_trajectory()
        if len(raw_traj) > 1:
            converted = convert_env_trajectory(raw_traj)
            collected_trajectories.append(converted)
        env.reset()
        return env.get_observation()

    @tool(description=(
        "Stop the exploration phase.  Call this when your current exploration "
        "plan is finished.  The current trajectory will be saved automatically."
    ))
    def stop_exploration() -> str:
        """Signal that exploration is complete."""
        raw_traj = env.get_raw_trajectory()
        if len(raw_traj) > 1:
            converted = convert_env_trajectory(raw_traj)
            collected_trajectories.append(converted)
        stop_flag["stopped"] = True
        return "Exploration stopped.  Your trajectories have been saved."

    return [env_step, env_reset, stop_exploration], stop_flag


# ============================================================
# Summarisation Prompt
# ============================================================

_SUMMARIZE_SYSTEM_PROMPT = """\
You are a scientific assistant.  You have just observed an exploration session \
where an agent interacted with a grid environment to discover its dynamics.

Your task is to summarise the exploration into structured Q&A pairs.  Each pair \
should capture one scientific question that was investigated and the answer or \
finding based on the observations.

Focus on concrete, factual findings about the environment's transition dynamics, \
for example:
- What happens when a specific action is taken?
- How do objects move or interact?
- Are there hidden states, counters, or modes?
- What are boundary behaviours?
"""

_SUMMARIZE_WITH_PENDING_SECTION = """
**The explorer was specifically asked to investigate these questions:**
{pending_questions}

Please make sure to include answers to these questions in your Q&A pairs.
"""

_SUMMARIZE_CONVERSATION_SECTION = """
**Full exploration conversation:**
{conversation}
"""


# ============================================================
# Question Generation Prompt (for free exploration)
# ============================================================

_GENERATE_QUESTIONS_PROMPT = """\
You are a scientist planning the next round of exploration in an interactive \
grid environment.  Your goal is to propose 2-3 specific, actionable scientific \
questions that should be investigated to better understand the environment's \
transition dynamics.

{context_section}

Based on what is known so far, either:
1. Propose 2-3 questions that would be most valuable to investigate next.  \
Each question should target a specific aspect of the dynamics that is not yet \
well understood.
2. If the current knowledge and code are already sufficient to fully describe \
the environment's transition dynamics and no further exploration would provide \
meaningful value, set `exploration_sufficient` to true.  In that case the \
questions list may be empty.
"""


# ============================================================
# Subgraph State
# ============================================================

class ExploreState(TypedDict):
    # Inputs (from MainState)
    current_code: Optional[str]
    explored_qa: list              # Historical Q&A pairs (List[dict])
    pending_questions: list        # Investigation questions (List[str]) — from refine or self-generated
    error_frames: list             # Mispredicted frames for context
    max_explore_steps: int
    env_name: str
    data_dir: str
    use_obfuscation: bool

    # Internal (not exposed to MainState)
    agent_messages: list           # Full ReAct agent conversation history

    # Outputs (read back by MainState)
    new_trajectories: list         # Trajectories collected this round
    new_qa: list                   # Q&A pairs produced this round
    questions_from_refine: bool    # True if questions came from refine, False if self-generated
    exploration_sufficient: bool   # True if LLM decides no more exploration is needed


# ============================================================
# Subgraph Builder
# ============================================================

def create_explore_graph(llm):
    """Build and compile the exploration subgraph.

    Flow::

        START -> question_router --(has questions)--> run_agent -> summarize -> END
                                 \\--(no questions)--> generate_explore_questions -> run_agent -> ...

    Args:
        llm: A LangChain chat model instance.

    Returns:
        A compiled LangGraph that accepts and returns ExploreState.
    """

    # ----------------------------------------------------------
    # Node: question router — set questions_from_refine flag
    # ----------------------------------------------------------
    def question_router_node(state: ExploreState) -> dict:
        """Set the questions_from_refine flag based on whether questions are
        pre-populated (from refine) or need to be generated."""
        has_questions = bool(state.get("pending_questions"))
        logger.info(
            "ExploreGraph: question_router — questions_from_refine=%s",
            has_questions,
        )
        return {"questions_from_refine": has_questions}

    def route_after_question_router(state: ExploreState) -> str:
        if state.get("questions_from_refine"):
            return "run_agent"
        return "generate_explore_questions"

    # ----------------------------------------------------------
    # Node: generate exploration questions (free explore path)
    # ----------------------------------------------------------
    def generate_explore_questions_node(state: ExploreState) -> dict:
        logger.info("ExploreGraph: generating exploration questions")

        context_parts: List[str] = []

        current_code = state.get("current_code")
        if current_code:
            context_parts.append(
                f"**Current predict_dynamics code:**\n"
                f"```python\n{current_code}\n```"
            )

        explored_qa = state.get("explored_qa", [])
        if explored_qa:
            qa_text = _format_qa_pairs(explored_qa)
            context_parts.append(
                f"**Scientific knowledge gathered so far:**\n{qa_text}"
            )

        if not context_parts:
            context_parts.append(
                "This is the first exploration round.  No code or prior "
                "knowledge exists yet.  Propose basic questions about the "
                "environment's dynamics."
            )

        context_section = "\n\n".join(context_parts)
        prompt = _GENERATE_QUESTIONS_PROMPT.format(
            context_section=context_section
        )

        exploration_sufficient = False
        try:
            structured_llm = llm.with_structured_output(ExplorationQuestions)
            result: ExplorationQuestions = structured_llm.invoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": (
                    "Propose questions for the next exploration round.  "
                    "If you believe the current knowledge is already "
                    "sufficient to fully describe the dynamics and no "
                    "further exploration would be valuable, set "
                    "exploration_sufficient to true."
                )},
            ])
            questions = [q.question for q in result.questions]
            exploration_sufficient = result.exploration_sufficient
        except Exception as e:
            logger.error("Question generation failed: %s", e)
            questions = []

        if exploration_sufficient:
            logger.info(
                "ExploreGraph: LLM decided exploration is sufficient — "
                "skipping further exploration."
            )
            return {
                "pending_questions": questions,
                "exploration_sufficient": True,
            }

        if not questions:
            questions = [
                "What are the basic dynamics of this environment?"
            ]

        logger.info(
            "ExploreGraph: generated %d exploration questions",
            len(questions),
        )

        return {
            "pending_questions": questions,
            "exploration_sufficient": False,
        }

    # ----------------------------------------------------------
    # Node: run the ReAct agent
    # ----------------------------------------------------------
    def run_agent_node(state: ExploreState) -> dict:
        logger.info("ExploreGraph: running ReAct agent")

        # Locate the .sexp program file
        data_dir = state["data_dir"]
        env_name = state["env_name"]
        for subdir in ["tests", "programs", ""]:
            candidate = os.path.join(data_dir, subdir, f"{env_name}.sexp")
            if os.path.isfile(candidate):
                program_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Cannot find {env_name}.sexp under {data_dir}"
            )

        env = ExplorationEnvironment(
            program_path=program_path,
            max_steps=state.get("max_explore_steps", 200),
            use_obfuscation=state.get("use_obfuscation", False),
        )
        env.reset()

        collected: List[Dict[str, Any]] = []
        tools, stop_flag = create_exploration_tools(env, collected)

        system_prompt = build_exploration_prompt(
            current_code=state.get("current_code"),
            explored_qa=state.get("explored_qa", []),
            pending_questions=state.get("pending_questions", []),
            error_frames=state.get("error_frames", []),
        )

        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

        initial_obs = env.get_observation()
        user_message = (
            "The environment has been initialized.  Here is the initial "
            f"observation:\n{initial_obs}\n\n"
            "Start exploring.  Remember to call `env_reset()` to collect "
            "multiple trajectories, and call `stop_exploration()` when done."
        )

        agent_messages: list = []
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_message}]},
            )
            agent_messages = result.get("messages", [])
        except Exception as e:
            logger.error("Exploration agent error: %s", e)
            # Save whatever trajectory we have
            raw_traj = env.get_raw_trajectory()
            if len(raw_traj) > 1:
                collected.append(convert_env_trajectory(raw_traj))

        logger.info(
            "ExploreGraph: agent collected %d trajectories, %d messages",
            len(collected), len(agent_messages),
        )

        return {
            "agent_messages": agent_messages,
            "new_trajectories": collected,
        }

    # ----------------------------------------------------------
    # Node: summarise exploration into structured Q&A
    # ----------------------------------------------------------
    def summarize_node(state: ExploreState) -> dict:
        logger.info("ExploreGraph: summarising exploration")

        agent_msgs = state.get("agent_messages") or []
        pending = state.get("pending_questions") or []

        # Build a text representation of the conversation for the summariser
        conversation_lines: List[str] = []
        for msg in agent_msgs:
            # Handle both LangChain message objects and dicts
            if hasattr(msg, "type"):
                role = msg.type  # "human", "ai", "tool"
            elif isinstance(msg, dict):
                role = msg.get("role", msg.get("type", "unknown"))
            else:
                role = "unknown"

            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", "")
            else:
                content = str(msg)

            # Truncate very long content (tool results with full state JSON)
            content_str = str(content)
            if len(content_str) > 2000:
                content_str = content_str[:2000] + "... [truncated]"

            conversation_lines.append(f"[{role}] {content_str}")

        conversation_text = "\n".join(conversation_lines)

        # Build summarisation prompt
        prompt = _SUMMARIZE_SYSTEM_PROMPT

        # pending_questions is always populated (either from refine or from
        # the generate_explore_questions node), so always include the section.
        ptext = "\n".join(f"- {q}" for q in pending)
        prompt += _SUMMARIZE_WITH_PENDING_SECTION.format(
            pending_questions=ptext)

        prompt += _SUMMARIZE_CONVERSATION_SECTION.format(
            conversation=conversation_text)

        # Use structured output
        try:
            structured_llm = llm.with_structured_output(ExplorationSummary)
            summary: ExplorationSummary = structured_llm.invoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": (
                    "Based on the exploration conversation above, produce "
                    "structured Q&A pairs summarising what was learned about "
                    "the environment dynamics."
                )},
            ])
            new_qa = [
                {"question": qa.question, "answer": qa.answer}
                for qa in summary.qa_pairs
            ]
        except Exception as e:
            logger.error("Summarisation failed: %s", e)
            new_qa = []

        logger.info("ExploreGraph: produced %d Q&A pairs", len(new_qa))

        return {
            "new_qa": new_qa,
        }

    # ----------------------------------------------------------
    # Routing helper: after generate_explore_questions
    # ----------------------------------------------------------
    def route_after_generate_questions(state: ExploreState) -> str:
        """If the LLM decided exploration is sufficient, skip straight to END."""
        if state.get("exploration_sufficient"):
            return END
        return "run_agent"

    # ----------------------------------------------------------
    # Build the graph
    # ----------------------------------------------------------
    workflow = StateGraph(ExploreState)

    workflow.add_node("question_router", question_router_node)
    workflow.add_node("generate_explore_questions", generate_explore_questions_node)
    workflow.add_node("run_agent", run_agent_node)
    workflow.add_node("summarize", summarize_node)

    workflow.set_entry_point("question_router")
    workflow.add_conditional_edges("question_router", route_after_question_router, {
        "run_agent": "run_agent",
        "generate_explore_questions": "generate_explore_questions",
    })
    workflow.add_conditional_edges("generate_explore_questions", route_after_generate_questions, {
        "run_agent": "run_agent",
        END: END,
    })
    workflow.add_edge("run_agent", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()
