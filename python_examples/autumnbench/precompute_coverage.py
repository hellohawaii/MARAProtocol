"""Replay study_v2 human trajectories on instrumented world models, write coverage_analysis_data.json.

Run from the project root:
    python -m MARAProtocol.python_examples.autumnbench.precompute_coverage

Or from this directory:
    python precompute_coverage.py
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
INSTRUMENTED_DIR = Path(__file__).resolve().parent / "example_benchmark" / "python_programs_instrumented"
TRAJECTORIES_DIR = REPO_ROOT / "AutumnWeb" / "user_study_trajectories" / "study_v2"
DEFAULT_OUTPUT = REPO_ROOT / "AutumnWeb" / "coverage_analysis_data.json"

ENVS = ["DQ8GC", "E3V6M", "EAHCW", "NRDF6", "NTQ4Y"]

ALL_RULES_PER_ENV: Dict[str, List[str]] = {
    "DQ8GC": [
        "on_adjacent_unhealthy",
        "on_clicked_swap",
        "movement_on_arrows",
    ],
    "E3V6M": [
        "on_clicked_toggle",
        "move_if_off_up",
        "move_if_off_down",
        "move_if_off_left",
        "move_if_off_right",
        "rotate_if_on_up",
        "rotate_if_on_down",
        "rotate_if_on_left",
        "rotate_if_on_right",
    ],
    "EAHCW": [
        "on_clicked_add_default_red",
        "on_clicked_add_colored",
        "arrow_up_set_gold",
        "arrow_down_set_purple",
        "arrow_left_set_green",
        "arrow_right_set_blue",
        "reset_down_when_prev_up",
        "reset_up_when_prev_down",
        "reset_left_when_prev_right",
        "reset_right_when_prev_left",
    ],
    "NTQ4Y": [
        "water_physics",
        "add_vessel",
        "add_plug",
        "add_water",
        "button_click_vessel",
        "button_click_plug",
        "button_click_water",
        "remove_plugs_button",
        "clear_all_button",
    ],
    "NRDF6": [
        "on_clicked_add_rock",
        "crate_weight_increase",
        "crate_weight_no_change",
    ],
}


def _add_helper_dir_to_syspath() -> None:
    helper = REPO_ROOT / "MARAProtocol" / "python_examples" / "autumnbench" / "explore_by_code" / "llm_workspace"
    if str(helper) not in sys.path:
        sys.path.insert(0, str(helper))


_add_helper_dir_to_syspath()
from check_traj_example import compare_visible_states  # noqa: E402


def _load_module(env_name: str) -> Tuple[Any, Any, set]:
    code_path = INSTRUMENTED_DIR / f"{env_name}.py"
    code_str = code_path.read_text(encoding="utf-8")
    exec_globals: Dict[str, Any] = {"__name__": f"_instrumented_{env_name}"}
    exec(compile(code_str, str(code_path), "exec"), exec_globals)
    init_fn = exec_globals.get("init_state")
    predict_fn = exec_globals.get("predict_dynamics")
    fired = exec_globals.get("FIRED")
    if not callable(init_fn) or not callable(predict_fn) or not isinstance(fired, set):
        raise RuntimeError(f"{env_name}: missing init_state/predict_dynamics/FIRED")
    return init_fn, predict_fn, fired


def _load_instructions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    prompts_dir = REPO_ROOT / "MARAProtocol" / "python_examples" / "autumnbench" / "example_benchmark" / "prompts"
    if not prompts_dir.exists():
        return out
    for env in ENVS:
        for suffix in ("_planning.json", "_planning_v2.json"):
            f = prompts_dir / f"{env}{suffix}"
            if f.exists():
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    instr = data.get("instruction") or data.get("planning_instruction") or data.get("prompt")
                    if isinstance(instr, str):
                        out[env] = instr
                        break
                except Exception:
                    pass
    return out


def _get_blacklisted_users(server_py: Path) -> List[str]:
    if not server_py.exists():
        return []
    text = server_py.read_text(encoding="utf-8")
    marker = "STUDY_V2_BLACKLISTED_USERS"
    idx = text.find(marker)
    if idx < 0:
        return []
    start = text.find("[", idx)
    end = text.find("]", start)
    if start < 0 or end < 0:
        return []
    raw = text[start : end + 1]
    try:
        parsed = json.loads(raw.replace("'", '"'))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return []


def _is_blacklisted(user_dir_name: str, blacklisted: List[str]) -> bool:
    username = user_dir_name.rsplit("_", 1)[0] if "_" in user_dir_name else user_dir_name
    return username in blacklisted


def _replay_episode(
    init_fn: Any,
    predict_fn: Any,
    fired_set: set,
    transitions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Single-step replay: at step i, feed rawFrame[i-1] (or our predicted state if missing)
    into predict_dynamics with action[i], compare predicted next state to rawFrame[i].
    Hidden state is propagated forward from init_state.
    """
    state, hidden = init_fn()

    fired_per_step: List[List[str]] = []
    matches: List[bool] = []
    is_noop: List[bool] = []

    if not transitions:
        return {"fired_per_step": [], "matches": [], "n_steps": 0, "is_noop": []}

    for i in range(1, len(transitions)):
        raw_action = transitions[i].get("action")
        action = raw_action or {"type": "noop"}
        action_type = action.get("type") if isinstance(action, dict) else None
        is_noop.append(raw_action is None or action_type == "noop")

        fired_set.clear()
        try:
            next_state, next_hidden = predict_fn(
                copy.deepcopy(state),
                copy.deepcopy(hidden),
                action,
            )
        except Exception:
            fired_per_step.append([])
            matches.append(False)
            actual_next = transitions[i].get("rawFrame")
            if isinstance(actual_next, dict) and actual_next:
                state = actual_next
            continue

        fired_per_step.append(sorted(fired_set))

        expected = transitions[i].get("rawFrame")
        matches.append(bool(compare_visible_states(next_state, expected)))

        if isinstance(expected, dict) and expected:
            state = expected
        else:
            state = next_state
        hidden = next_hidden

    return {
        "fired_per_step": fired_per_step,
        "matches": matches,
        "n_steps": len(fired_per_step),
        "is_noop": is_noop,
    }


def _process_env(env_name: str, blacklisted: List[str]) -> List[Dict[str, Any]]:
    print(f"[{env_name}] loading instrumented module...", flush=True)
    init_fn, predict_fn, fired_set = _load_module(env_name)
    rules = ALL_RULES_PER_ENV[env_name]
    denom = max(1, len(rules))

    env_dir = TRAJECTORIES_DIR / env_name
    if not env_dir.exists():
        print(f"  no trajectories at {env_dir}", flush=True)
        return []

    user_curves: List[Dict[str, Any]] = []

    for user_dir in sorted(d for d in env_dir.iterdir() if d.is_dir()):
        if _is_blacklisted(user_dir.name, blacklisted):
            continue

        episodes: List[Tuple[str, Dict[str, Any]]] = []
        for f in sorted(user_dir.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  skipping {f.name}: {e}", flush=True)
                continue
            if data.get("env_name") != env_name:
                continue
            episodes.append((f.name, data))

        if not episodes:
            continue

        episodes.sort(key=lambda x: x[1].get("timestamp", x[0]))

        username = episodes[0][1].get("username", "unknown")
        seen_rules: set[str] = set()
        coverage_curve: List[float] = []
        is_noop_per_step: List[bool] = []
        rules_first_seen: Dict[str, int] = {}
        episode_boundaries: List[int] = [0]
        episode_meta: List[Dict[str, Any]] = []
        global_step = 0
        total_correct = 0
        total_compared = 0

        for fname, data in episodes:
            transitions = data.get("trajectory", [])
            ep = _replay_episode(init_fn, predict_fn, fired_set, transitions)

            for idx, fired_list in enumerate(ep["fired_per_step"]):
                for rule in fired_list:
                    if rule not in rules_first_seen:
                        rules_first_seen[rule] = global_step
                seen_rules.update(fired_list)
                coverage_curve.append(len(seen_rules) / denom)
                is_noop_per_step.append(ep["is_noop"][idx] if idx < len(ep["is_noop"]) else False)
                global_step += 1

            total_compared += len(ep["matches"])
            total_correct += sum(1 for m in ep["matches"] if m)

            ep_acc = (
                sum(1 for m in ep["matches"] if m) / len(ep["matches"])
                if ep["matches"] else 0.0
            )
            episode_meta.append(
                {
                    "file": fname,
                    "instruction_shown": bool(data.get("instruction_shown", False)),
                    "goal_reached": bool(data.get("goal_reached", False)),
                    "reason": data.get("reason"),
                    "num_steps": ep["n_steps"],
                    "match_accuracy": ep_acc,
                }
            )
            episode_boundaries.append(global_step)

        instruction_shown_session = bool(episode_meta[0]["instruction_shown"]) if episode_meta else False

        user_curves.append(
            {
                "env_name": env_name,
                "username": username,
                "user_key": user_dir.name,
                "instruction_shown": instruction_shown_session,
                "coverage_curve": coverage_curve,
                "is_noop_per_step": is_noop_per_step,
                "rules_first_seen": rules_first_seen,
                "episode_boundaries": episode_boundaries,
                "episodes": episode_meta,
                "prediction_accuracy": (total_correct / total_compared) if total_compared else 0.0,
            }
        )

    return user_curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", help="Restrict to one or more envs (default: all five)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    args = parser.parse_args()

    envs = args.env if args.env else ENVS
    invalid = [e for e in envs if e not in ENVS]
    if invalid:
        print(f"Unknown envs: {invalid}. Valid: {ENVS}", file=sys.stderr)
        return 2

    blacklisted = _get_blacklisted_users(REPO_ROOT / "AutumnWeb" / "server.py")
    if blacklisted:
        print(f"Blacklisted users: {blacklisted}", flush=True)

    instructions = _load_instructions()

    user_curves: List[Dict[str, Any]] = []
    for env in envs:
        user_curves.extend(_process_env(env, blacklisted))

    output = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "env_names": envs,
        "all_rules_per_env": {env: ALL_RULES_PER_ENV[env] for env in envs},
        "instructions": {k: v for k, v in instructions.items() if k in envs},
        "blacklisted_users": blacklisted,
        "user_curves": user_curves,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nWrote {args.out}")
    print(f"\n{'env':<8}{'n_users':<10}{'mean_acc':<12}{'rules_used':<12}")
    print("-" * 50)
    for env in envs:
        env_curves = [c for c in user_curves if c["env_name"] == env]
        n_users = len(env_curves)
        mean_acc = (
            sum(c["prediction_accuracy"] for c in env_curves) / n_users
            if n_users else 0.0
        )
        used_rules: set[str] = set()
        for c in env_curves:
            used_rules.update(c["rules_first_seen"].keys())
        total_rules = len(ALL_RULES_PER_ENV[env])
        print(f"{env:<8}{n_users:<10}{mean_acc:<12.3f}{len(used_rules)} / {total_rules}")
        for c in env_curves:
            if c["prediction_accuracy"] < 0.9:
                print(f"  ! {c['user_key']:<35}acc={c['prediction_accuracy']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
