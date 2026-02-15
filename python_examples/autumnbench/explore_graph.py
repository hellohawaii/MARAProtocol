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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
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

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, BaseMessage):
        return {
            "type": getattr(obj, "type", obj.__class__.__name__),
            "content": getattr(obj, "content", ""),
        }
    return obj


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)


def _log_llm_call(explore_log_dir: Optional[str], node_name: str,
                  llm_input: Any, llm_output: Any, error: Optional[str] = None) -> None:
    if not explore_log_dir:
        return
    ts = int(time.time() * 1000)
    out_path = os.path.join(
        explore_log_dir, "llm_calls", f"{ts}_{node_name}.json")
    _write_json(out_path, {
        "node": node_name,
        "llm_input": llm_input,
        "llm_output": llm_output,
        "error": error,
    })


class _JsonTraceCallback(BaseCallbackHandler):
    """Lightweight callback to persist low-level LLM request/response traces."""

    def __init__(self, explore_log_dir: Optional[str]):
        self.explore_log_dir = explore_log_dir

    def _write_event(self, event: str, run_id: Any, payload: Dict[str, Any]) -> None:
        if not self.explore_log_dir:
            return
        ts = int(time.time() * 1000)
        rid = str(run_id).replace("-", "")
        out_path = os.path.join(
            self.explore_log_dir, "llm_traces", f"{ts}_{rid}_{event}.json")
        _write_json(out_path, payload)

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]],
                            run_id: Any, **kwargs: Any) -> Any:
        self._write_event("chat_start", run_id, {
            "serialized": serialized,
            "messages": messages,
        })

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
                     run_id: Any, **kwargs: Any) -> Any:
        self._write_event("llm_start", run_id, {
            "serialized": serialized,
            "prompts": prompts,
        })

    def on_llm_end(self, response: Any, run_id: Any, **kwargs: Any) -> Any:
        generations: List[str] = []
        for group in getattr(response, "generations", []) or []:
            for gen in group:
                text = getattr(gen, "text", None)
                if text is None and hasattr(gen, "message"):
                    text = getattr(gen.message, "content", None)
                generations.append("" if text is None else str(text))
        self._write_event("llm_end", run_id, {
            "generations": generations,
            "llm_output": getattr(response, "llm_output", {}),
        })


_BASIC_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (220, 20, 60),
    "blue": (65, 105, 225),
    "green": (50, 205, 50),
    "yellow": (255, 215, 0),
    "orange": (255, 140, 0),
    "purple": (138, 43, 226),
    "pink": (255, 105, 180),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "slategrey": (112, 128, 144),
    "slategray": (112, 128, 144),
}


def _color_to_rgb(color_name: Any) -> tuple:
    if isinstance(color_name, str):
        c = color_name.strip().lower()
        if c in _BASIC_COLORS:
            return _BASIC_COLORS[c]
        if c.startswith("#") and len(c) == 7:
            try:
                return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16))
            except Exception:
                pass
    seed = abs(hash(str(color_name))) % (256 ** 3)
    return ((seed >> 16) & 255, (seed >> 8) & 255, seed & 255)


def _state_to_grid(state: Dict[str, Any]) -> Optional[List[List[str]]]:
    gs = state.get("GRID_SIZE")
    if not isinstance(gs, int) or gs <= 0:
        return None
    grid = [["black"] * gs for _ in range(gs)]
    keys = sorted(k for k in state.keys() if k != "GRID_SIZE")
    if "background" in keys:
        keys.insert(0, keys.pop(keys.index("background")))
    for key in keys:
        objects = state.get(key)
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            pos = obj.get("position")
            if not isinstance(pos, dict):
                continue
            x = pos.get("x")
            y = pos.get("y")
            if not isinstance(x, int) or not isinstance(y, int):
                continue
            if not (0 <= y < gs and 0 <= x < gs):
                continue
            color = obj.get("color", key)
            grid[y][x] = str(color)
    return grid


def _render_grid_png(grid: List[List[str]], out_path: str, cell_size: int = 24) -> None:
    if Image is None or ImageDraw is None:
        return
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h == 0 or w == 0:
        return
    img = Image.new("RGB", (w * cell_size, h * cell_size), "black")
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, color_name in enumerate(row):
            rgb = _color_to_rgb(color_name)
            x0, y0 = x * cell_size, y * cell_size
            x1, y1 = x0 + cell_size - 1, y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=rgb)
            draw.rectangle([x0, y0, x1, y1], outline=(40, 40, 40))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def _save_trajectory_video_like_artifacts(
    trajectories: List[Dict[str, Any]],
    explore_log_dir: Optional[str],
) -> None:
    if not explore_log_dir or Image is None:
        return
    videos_root = os.path.join(explore_log_dir, "videos")
    os.makedirs(videos_root, exist_ok=True)
    for traj_idx, traj in enumerate(trajectories):
        traj_dir = os.path.join(videos_root, f"trajectory_{traj_idx:03d}")
        frames_dir = os.path.join(traj_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths: List[str] = []
        frames = traj.get("frames", [])
        for frame_idx, frame in enumerate(frames):
            raw_state = frame.get("rawFrame", {})
            grid = _state_to_grid(raw_state)
            if grid is None:
                continue
            png_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
            _render_grid_png(grid, png_path)
            if os.path.isfile(png_path):
                frame_paths.append(png_path)
        if not frame_paths:
            continue
        gif_path = os.path.join(traj_dir, "trajectory.gif")
        try:
            images = [Image.open(p).convert("P") for p in frame_paths]
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=300,
                loop=0,
            )
        except Exception as e:
            logger.warning("Failed to create trajectory gif for %s: %s", traj_dir, e)


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
        # Keep `time` as a global budget counter across resets within one
        # exploration run (run_agent_node). Do not reset it here.
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

This environment is a deterministic grid world: it consists of a grid containing
cells that can take colors. The grid has width GRID_SIZE and height GRID_SIZE
(i.e., a GRID_SIZE x GRID_SIZE grid). There is no randomness: the same state
and action always produce the same next state.

**Environment action space (valid actions):**
- `click x y` - Click on the cell at location (x, y). For GRID_SIZE, x and y
  must each be between 0 and GRID_SIZE-1 inclusive.
- `left` - Press the left arrow key.
- `right` - Press the right arrow key.
- `up` - Press the up arrow key.
- `down` - Press the down arrow key.
- `noop` - Do nothing and continue to the next step.

**Available tools:**
- `env_step(action)` — execute one environment action from the action space above.
- `env_reset()` — reset the environment and start collecting a new trajectory.  \
Use this to gather multiple independent trajectories.
- `stop_exploration()` — call this when you have completed your current \
exploration plan.

Important: `env_reset()` and `stop_exploration()` are tools for trajectory
management; they are not part of the environment action space.

**When to stop:**
You do NOT need to collect exhaustively complete data before stopping.  This is \
an iterative process — you will be called again later to explore further.  Stop when:
- You have collected enough evidence to answer the `pending_questions` listed below.
- If there are no `pending_questions`, stop when your current exploration plan is finished.

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

_EXPLORE_SELF_GENERATED_SECTION = """
**IMPORTANT — Self-generated investigation questions for this exploration round:**
These questions were generated to guide free exploration and improve understanding \
of the environment dynamics.  Please design your exploration to collect trajectories \
that answer these questions.

{questions}
"""

_EXPLORE_ERRORS_SECTION = """
**Frames where the current code makes wrong predictions (for context):**
{errors}
"""


def _format_qa_pairs(qa_list: List[Dict[str, Any]]) -> str:
    """Format a list of Q&A dicts into readable text."""
    if not qa_list:
        return "(none yet)"
    parts: List[str] = [
        "(`confidence` is in [0,1]. Higher means the QA is more reliable; "
        "0 means the QA is very likely incorrect and should generally not be relied on.)",
        "",
    ]
    for i, qa in enumerate(qa_list, 1):
        conf = qa.get("confidence")
        conf_text = f" [confidence={float(conf):.3f}]" if isinstance(conf, (int, float)) else ""
        parts.append(f"Q{i}: {qa.get('question', '?')}")
        parts.append(f"A{i}{conf_text}: {qa.get('answer', '?')}")
        parts.append("")
    return "\n".join(parts)


def build_exploration_prompt(
    current_code: Optional[str],
    explored_qa: List[Dict[str, Any]],
    pending_questions: List[str],
    error_frames: List[Dict[str, Any]],
    questions_from_refine: bool,
) -> str:
    """Build the system prompt for the exploration agent."""
    prompt = _EXPLORE_SYSTEM_PROMPT

    if current_code:
        prompt += _EXPLORE_CODE_SECTION.format(code=current_code)

    if explored_qa:
        prompt += _EXPLORE_QA_SECTION.format(qa_text=_format_qa_pairs(explored_qa))

    if pending_questions:
        qtext = "\n".join(f"- {q}" for q in pending_questions)
        if questions_from_refine:
            prompt += _EXPLORE_PENDING_SECTION.format(questions=qtext)
        else:
            prompt += _EXPLORE_SELF_GENERATED_SECTION.format(questions=qtext)

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
    ``{"stopped": bool, "reason": Optional[str]}`` mutated by tools.
    """
    stop_flag = {"stopped": False, "reason": None}

    @tool(description=(
        "Take an action in the environment. "
        "Valid actions: 'click x y', 'left', 'right', 'up', 'down', 'noop'. "
        "Returns the observation after the action."
    ))
    def env_step(action: str) -> str:
        """Take an action in the environment and return the new observation."""
        try:
            if env.time >= env.max_steps:
                stop_flag["stopped"] = True
                stop_flag["reason"] = "max_steps"
                return (
                    f"Max steps ({env.max_steps}) reached. "
                    "Exploration must stop now."
                )
            obs = env.step(action)
            if env.time >= env.max_steps:
                # Step budget is exhausted after this transition. Signal the outer
                # stream loop to stop regardless of what the model does next.
                stop_flag["stopped"] = True
                stop_flag["reason"] = "max_steps"
            return obs
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
        stop_flag["reason"] = "agent_stop"
        return "Exploration stopped.  Your trajectories have been saved."

    return [env_step, env_reset, stop_exploration], stop_flag


# ============================================================
# Summarisation Prompt
# ============================================================

_SUMMARIZE_SYSTEM_PROMPT = """\
You are a scientific assistant.  You have just observed an exploration session \
where an agent interacted with a grid environment to discover its dynamics.

Your task is to answer the \
provided pending investigation questions, using evidence from the exploration \
conversation and trajectories.

Output structured Q&A pairs where:
- Each Q is one pending investigation question (use the same wording).
- Each A is a concrete, factual answer grounded in observed behavior.
- Do not add extra questions beyond the pending list.
"""

_SUMMARIZE_CODE_SECTION = """
**Current predict_dynamics code (for reference):**
```python
{code}
```
"""

_SUMMARIZE_WITH_PENDING_FROM_REFINE_SECTION = """
**These investigation questions came from the code-generation/refinement stage:**
The current code could not explain all observed trajectories.  Therefore, the
code-generation/refinement process raised the following unresolved questions:
{pending_questions}

Please return Q&A pairs for these questions only.
"""

_SUMMARIZE_WITH_PENDING_SELF_GENERATED_SECTION = """
**These investigation questions were self-generated for this free-exploration round:**
The current code can explain previously collected trajectories, but it is still
unclear whether it also explains the newly collected trajectories from this
round. You may refer to the current code for context. Use the latest exploration evidence to answer the following questions:
{pending_questions}

Please return Q&A pairs for these questions only.
"""

# ============================================================
# Question Generation Prompt (for free exploration)
# ============================================================

_GENERATE_QUESTIONS_PROMPT = """\
You are a scientist planning the next round of exploration in an interactive \
deterministic grid world environment.  Your goal is to propose 1 specific, actionable scientific \
questions that should be investigated to better understand the environment's \
transition dynamics.

The environment consists of a grid containing cells that can take colors. The
grid has width GRID_SIZE and height GRID_SIZE (i.e., a GRID_SIZE x GRID_SIZE
grid). The dynamics are deterministic: given the same state and action, the
next state is always the same (no randomness).

The valid environment actions are:
- `click x y` - Click on the cell at location (x, y). For GRID_SIZE, x and y
  must each be between 0 and GRID_SIZE-1 inclusive.
- `left` - Press the left arrow key.
- `right` - Press the right arrow key.
- `up` - Press the up arrow key.
- `down` - Press the down arrow key.
- `noop` - Do nothing and continue to the next step.

Important: `reset` and `stop_exploration` are tools for trajectory management,
not environment actions in the action space.

{context_section}

Based on what is known so far, either:
1. Propose exactly 1 question that would be most valuable to investigate next.  \
The question should target a specific aspect of the dynamics that is not yet \
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
    explore_log_dir: str

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

        # Include the initial observation so question generation can anchor on
        # the concrete starting state, before proposing exploration targets.
        try:
            data_dir = state["data_dir"]
            env_name = state["env_name"]
            for subdir in ["tests", "programs", ""]:
                candidate = os.path.join(data_dir, subdir, f"{env_name}.sexp")
                if os.path.isfile(candidate):
                    program_path = candidate
                    break
            else:
                program_path = None

            if program_path:
                init_env = ExplorationEnvironment(
                    program_path=program_path,
                    max_steps=state.get("max_explore_steps", 200),
                    use_obfuscation=state.get("use_obfuscation", False),
                )
                init_env.reset()
                initial_obs = init_env.get_observation()
                context_parts.append(
                    f"**Initial observation:**\n```json\n{initial_obs}\n```"
                )
        except Exception as e:
            logger.warning(
                "ExploreGraph: failed to fetch initial observation for question generation: %s",
                e,
            )

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
        llm_messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": (
                "Propose questions for the next exploration round.  "
                "If you believe the current knowledge is already "
                "sufficient to fully describe the dynamics and no "
                "further exploration would be valuable, set "
                "exploration_sufficient to true."
            )},
        ]
        try:
            structured_llm = llm.with_structured_output(ExplorationQuestions)
            trace_cb = _JsonTraceCallback(state.get("explore_log_dir"))
            result: ExplorationQuestions = structured_llm.invoke(
                llm_messages, config={"callbacks": [trace_cb]})
            questions = [q.question for q in result.questions]
            exploration_sufficient = result.exploration_sufficient
            _log_llm_call(
                state.get("explore_log_dir"),
                "generate_explore_questions",
                llm_messages,
                {
                    "exploration_sufficient": exploration_sufficient,
                    "questions": questions,
                },
            )
        except Exception as e:
            logger.error("Question generation failed: %s", e)
            questions = []
            _log_llm_call(
                state.get("explore_log_dir"),
                "generate_explore_questions",
                llm_messages,
                {},
                error=str(e),
            )

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
            questions_from_refine=state.get("questions_from_refine", False),
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
            trace_cb = _JsonTraceCallback(state.get("explore_log_dir"))
            for event in agent.stream(
                {"messages": [{"role": "user", "content": user_message}]},
                config={
                    "callbacks": [trace_cb],
                },
                stream_mode="values",
            ):
                if isinstance(event, dict) and "messages" in event:
                    agent_messages = event.get("messages", []) or agent_messages
                if stop_flag["stopped"]:
                    break

            # If we force-stopped due to budget exhaustion, the current trajectory
            # may not have been saved via env_reset/stop_exploration yet.
            if stop_flag.get("reason") == "max_steps":
                raw_traj = env.get_raw_trajectory()
                if len(raw_traj) > 1:
                    collected.append(convert_env_trajectory(raw_traj))

            _log_llm_call(
                state.get("explore_log_dir"),
                "run_agent",
                {
                    "system_prompt": system_prompt,
                    "initial_user_message": user_message,
                },
                {
                    "agent_messages": agent_messages,
                    "num_trajectories": len(collected),
                    "stop_reason": stop_flag.get("reason"),
                },
            )
        except Exception as e:
            logger.error("Exploration agent error: %s", e)
            # Save whatever trajectory we have
            raw_traj = env.get_raw_trajectory()
            if len(raw_traj) > 1:
                collected.append(convert_env_trajectory(raw_traj))
            _log_llm_call(
                state.get("explore_log_dir"),
                "run_agent",
                {
                    "system_prompt": system_prompt,
                    "initial_user_message": user_message,
                },
                {},
                error=str(e),
            )

        if state.get("explore_log_dir"):
            _write_json(
                os.path.join(state["explore_log_dir"], "collected_trajectories.json"),
                {
                    "num_trajectories": len(collected),
                    "trajectories": collected,
                },
            )
            _save_trajectory_video_like_artifacts(
                collected, state.get("explore_log_dir"))

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
        current_code = state.get("current_code")

        # Build summarisation prompt
        summarize_system_prompt = _SUMMARIZE_SYSTEM_PROMPT
        if current_code:
            summarize_system_prompt += _SUMMARIZE_CODE_SECTION.format(
                code=current_code
            )

        # pending_questions is always populated (either from refine or from
        # the generate_explore_questions node), so always include the section.
        ptext = "\n".join(f"- {q}" for q in pending)
        questions_from_refine = state.get("questions_from_refine", False)
        if questions_from_refine:
            summarize_system_prompt += _SUMMARIZE_WITH_PENDING_FROM_REFINE_SECTION.format(
                pending_questions=ptext
            )
            summarize_user_prompt = (
                "Answer the pending investigation questions from the "
                "code-generation/refinement stage using the exploration "
                "conversation above. Return structured Q&A pairs for those "
                "pending questions only."
            )
        else:
            summarize_system_prompt += _SUMMARIZE_WITH_PENDING_SELF_GENERATED_SECTION.format(
                pending_questions=ptext
            )
            summarize_user_prompt = (
                "Answer the pending self-generated investigation questions "
                "for this exploration round using the exploration conversation "
                "above. Return structured Q&A pairs for those pending "
                "questions only."
            )

        # Use structured output
        try:
            structured_llm = llm.with_structured_output(ExplorationSummary)
            llm_messages: List[Any] = [{"role": "system",
                                        "content": summarize_system_prompt}]
            llm_messages.extend(agent_msgs)
            llm_messages.append({"role": "user", "content": summarize_user_prompt})
            trace_cb = _JsonTraceCallback(state.get("explore_log_dir"))
            summary: ExplorationSummary = structured_llm.invoke(
                llm_messages, config={"callbacks": [trace_cb]})
            new_qa = [
                {
                    "question": qa.question,
                    "answer": qa.answer,
                    "confidence": 0.7,
                    "alpha": 1.4,
                    "beta": 0.6,
                    "confidence_history": [],
                }
                for qa in summary.qa_pairs
            ]
            _log_llm_call(
                state.get("explore_log_dir"),
                "summarize",
                llm_messages,
                {"new_qa": new_qa},
            )
        except Exception as e:
            logger.error("Summarisation failed: %s", e)
            new_qa = []
            _log_llm_call(
                state.get("explore_log_dir"),
                "summarize",
                {
                    "prompt": summarize_system_prompt,
                    "agent_messages": agent_msgs,
                    "pending_questions": pending,
                },
                {},
                error=str(e),
            )

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
