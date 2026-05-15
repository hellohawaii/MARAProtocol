"""Replay study_v2 trajectories and emit per-(env, condition) exploration graphs.

Output: AutumnWeb/exploration_graph_data.json — independent of coverage_analysis_data.json.

Run:
    python precompute_exploration_graph.py
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from precompute_coverage import (
    ENVS,
    REPO_ROOT,
    TRAJECTORIES_DIR,
    _get_blacklisted_users,
    _is_blacklisted,
    _load_instructions,
    _load_module,
)


DEFAULT_OUTPUT = REPO_ROOT / "AutumnWeb" / "exploration_graph_data.json"

ACTION_TYPES = ("click", "up", "down", "left", "right")


def _abstract_state_DQ8GC(_hidden: Dict[str, Any]) -> str:
    return "default"


def _abstract_state_E3V6M(hidden: Dict[str, Any]) -> str:
    return "on" if bool(hidden.get("turnedOn", False)) else "off"


def _abstract_state_EAHCW(hidden: Dict[str, Any]) -> str:
    val = str(hidden.get("active_arrow", "none"))
    if val not in {"none", "up", "down", "left", "right"}:
        val = "none"
    return val


def _abstract_state_NTQ4Y(hidden: Dict[str, Any]) -> str:
    val = str(hidden.get("currentParticle", "vessel"))
    if val not in {"vessel", "plug", "water"}:
        val = "vessel"
    return val


def _abstract_state_NRDF6(hidden: Dict[str, Any]) -> str:
    weight = int(hidden.get("crate_add_weight", 0))
    return "weight=5" if weight >= 5 else "weight<5"


ABSTRACT_STATE_LABEL_FNS: Dict[str, Callable[[Dict[str, Any]], str]] = {
    "DQ8GC": _abstract_state_DQ8GC,
    "E3V6M": _abstract_state_E3V6M,
    "EAHCW": _abstract_state_EAHCW,
    "NTQ4Y": _abstract_state_NTQ4Y,
    "NRDF6": _abstract_state_NRDF6,
}

NODE_ORDER: Dict[str, List[str]] = {
    "DQ8GC": ["default"],
    "E3V6M": ["off", "on"],
    "EAHCW": ["none", "up", "down", "left", "right"],
    "NTQ4Y": ["vessel", "plug", "water"],
    "NRDF6": ["weight<5", "weight=5"],
}


def _process_user_episodes(
    env_name: str,
    init_fn: Any,
    predict_fn: Any,
    fired_set: set,
    episodes: List[Tuple[str, Dict[str, Any]]],
    edge_counts: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> None:
    """Replay one user's episodes (already sorted) and update edge_counts in place.

    edge_counts entries shape: {"count": int, "unlocked_rules": set[str]}.
    """
    label_fn = ABSTRACT_STATE_LABEL_FNS[env_name]

    seen_rules: set[str] = set()

    for _fname, data in episodes:
        transitions = data.get("trajectory", [])
        state, hidden = init_fn()

        for i in range(1, len(transitions)):
            raw_action = transitions[i].get("action")
            if raw_action is None:
                action_type = "noop"
            else:
                action_type = (
                    raw_action.get("type") if isinstance(raw_action, dict) else None
                )

            if action_type not in ACTION_TYPES:
                # noop: advance state but do NOT update seen_rules, so always-on
                # rules (e.g. on_adjacent_unhealthy, water_physics) get attributed
                # to the next real user action edge instead of disappearing.
                action_for_call = raw_action or {"type": "noop"}
                fired_set.clear()
                try:
                    next_state, next_hidden = predict_fn(
                        copy.deepcopy(state),
                        copy.deepcopy(hidden),
                        action_for_call,
                    )
                except Exception:
                    actual = transitions[i].get("rawFrame")
                    if isinstance(actual, dict) and actual:
                        state = actual
                    continue
                actual = transitions[i].get("rawFrame")
                state = actual if isinstance(actual, dict) and actual else next_state
                hidden = next_hidden
                continue

            src_state = label_fn(hidden)

            fired_set.clear()
            try:
                next_state, next_hidden = predict_fn(
                    copy.deepcopy(state),
                    copy.deepcopy(hidden),
                    raw_action,
                )
            except Exception:
                actual = transitions[i].get("rawFrame")
                if isinstance(actual, dict) and actual:
                    state = actual
                continue

            fired_now = set(fired_set)
            unlocked_now = fired_now - seen_rules
            seen_rules |= fired_now

            dst_state = label_fn(next_hidden)

            key = (src_state, action_type, dst_state)
            entry = edge_counts.setdefault(key, {"count": 0, "unlocked_rules": set()})
            entry["count"] += 1
            entry["unlocked_rules"] |= unlocked_now

            actual = transitions[i].get("rawFrame")
            state = actual if isinstance(actual, dict) and actual else next_state
            hidden = next_hidden


def _process_env(env_name: str, blacklisted: List[str]) -> Dict[str, Dict[str, Any]]:
    print(f"[{env_name}] loading instrumented module...", flush=True)
    init_fn, predict_fn, fired_set = _load_module(env_name)

    env_dir = TRAJECTORIES_DIR / env_name
    if not env_dir.exists():
        print(f"  no trajectories at {env_dir}", flush=True)
        empty = {"nodes": NODE_ORDER[env_name], "edges": [], "n_users": 0}
        return {"with_instruction": empty, "no_instruction": dict(empty)}

    edges_by_cond: Dict[bool, Dict[Tuple[str, str, str], Dict[str, Any]]] = {
        True: {},
        False: {},
    }
    n_users_by_cond: Dict[bool, int] = {True: 0, False: 0}

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
        instruction_shown = bool(episodes[0][1].get("instruction_shown", False))
        n_users_by_cond[instruction_shown] += 1

        _process_user_episodes(
            env_name,
            init_fn,
            predict_fn,
            fired_set,
            episodes,
            edges_by_cond[instruction_shown],
        )

    out: Dict[str, Dict[str, Any]] = {}
    for cond_bool, cond_label in ((True, "with_instruction"), (False, "no_instruction")):
        edge_list: List[Dict[str, Any]] = []
        for (src, action, dst), counts in sorted(edges_by_cond[cond_bool].items()):
            edge_list.append(
                {
                    "src": src,
                    "action": action,
                    "dst": dst,
                    "count": counts["count"],
                    "unlocked_rules": sorted(counts["unlocked_rules"]),
                }
            )
        out[cond_label] = {
            "nodes": NODE_ORDER[env_name],
            "edges": edge_list,
            "n_users": n_users_by_cond[cond_bool],
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", action="append", help="Restrict to one or more envs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
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

    graphs: Dict[str, Dict[str, Any]] = {}
    for env in envs:
        graphs[env] = _process_env(env, blacklisted)

    output = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "env_names": envs,
        "instructions": {k: v for k, v in instructions.items() if k in envs},
        "blacklisted_users": blacklisted,
        "graphs": graphs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}\n")

    print(f"{'env':<8}{'with':<8}{'no':<8}{'nodes':<8}{'edges_with':<14}{'edges_no':<10}")
    print("-" * 60)
    for env in envs:
        g = graphs[env]
        print(
            f"{env:<8}"
            f"{g['with_instruction']['n_users']:<8}"
            f"{g['no_instruction']['n_users']:<8}"
            f"{len(NODE_ORDER[env]):<8}"
            f"{len(g['with_instruction']['edges']):<14}"
            f"{len(g['no_instruction']['edges']):<10}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
