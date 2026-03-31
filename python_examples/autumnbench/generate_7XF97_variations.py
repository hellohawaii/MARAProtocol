#!/usr/bin/env python3
"""Generate 1000 variations of the 7XF97 environment with different initial/target states.

Each variation has identical transition functions but different:
- Sun x-position and direction
- Cloud x-position
- Leaf column positions, initial heights, and target heights
- Mask (which changed leaf columns are evaluated)

Verification is done by loading each .sexp into the real Interpreter and running a policy.
"""

import copy
import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "explore_by_code"))

from interpreter_module import Interpreter
from autumnstdlib import autumnstdlib
from planning_utils import load_color_dict, check_grid_same

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE = 16
EXAMPLE_BENCHMARK_DIR = SCRIPT_DIR / "example_benchmark"
OUTPUT_DIR = EXAMPLE_BENCHMARK_DIR / "program_with_testcases" / "7XF97"
COLOR_DICT_PATH = EXAMPLE_BENCHMARK_DIR / "color_dict.yaml"

# Color integer mapping (from color_dict.yaml)
COLOR_INT = {
    "black": 1, "gray": 2, "gold": 4, "green": 5,
    "mediumpurple": 6, "blue": 10,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class VariationParams:
    sun_x: int
    cloud_x: int
    sun_moving_left: bool
    leaf_xs: List[int]
    leaf_heights_init: List[int]
    leaf_heights_target: List[int]
    mask_leaf_indices: List[int]

    def param_hash(self) -> str:
        payload = repr((
            self.sun_x, self.cloud_x, self.sun_moving_left,
            self.leaf_xs, self.leaf_heights_init,
            self.leaf_heights_target, self.mask_leaf_indices,
        ))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# .sexp generation
# ---------------------------------------------------------------------------
SEXP_TEMPLATE = """\
(program
  (= GRID_SIZE 16)

  (object Water (Cell 0 0 "blue"))
  (object Leaf (: color String) (Cell 0 0 color))
  (object Cloud (list (Cell -1 0 "gray") (Cell 0 0 "gray") (Cell 1 0 "gray") (Cell 2 0 "gray")
                      (Cell -1 1 "gray") (Cell 0 1 "gray") (Cell 1 1 "gray") (Cell 2 1 "gray")
                      (Cell -1 2 "gray") (Cell 0 2 "gray") (Cell 1 2 "gray") (Cell 2 2 "gray")))

  (object Sun (: movingLeft Bool) (list (Cell 0 0 "gold")
                                        (Cell 0 1 "gold")
                                        (Cell 1 0 "gold")
                                        (Cell 1 1 "gold")
                                        (Cell 0 2 "gold")
                                        (Cell 1 2 "gold")
                                        (Cell 2 0 "gold")
                                        (Cell 2 1 "gold")
                                        (Cell 2 2 "gold")))

  (: sun Sun)
  (= sun (initnext {sun_init} (prev sun)))

  (: water (List Water))
  (= water (initnext (list) (filter isWithinBounds (map (--> o (moveDown o)) (prev water))) ))

  (: cloud Cloud)
  (= cloud (initnext {cloud_init} (prev cloud)))

  (: leaves (List Leaf))
  (= leaves (initnext {leaves_init} (prev leaves)))

  (on down
    (= water (addObj (prev water) (Water (.. (moveDown (prev cloud)) origin)))))

  (on (intersects (map (--> obj (moveDown obj)) (prev water) ) (prev leaves))
    (= water (filter (--> obj (!(intersects (moveDown obj) (prev leaves)))) water) ) )

  (on (and (intersects (map moveDown (prev water)) (filter (--> obj (== (.. obj color) "green")) (prev leaves))) (! (intersects (prev sun) (prev cloud))))
    (= leaves (addObj (prev leaves) (map (--> obj (Leaf (if (== (.. (.. (moveUp obj) origin) y) 12) then "mediumpurple" else "green") (.. (moveUp obj) origin))) (filter (--> obj (intersects (moveUp obj) (prev water))) (prev leaves))))))

  (on left (= cloud (moveLeft (prev cloud))))
  (on right (= cloud (moveRight (prev cloud))))

  (on (== (.. (.. (prev sun) origin) x) 0) (= sun (updateObj (prev sun) "movingLeft" false)))
  (on (== (.. (.. (prev sun) origin) x) (- GRID_SIZE 3)) (= sun (updateObj (prev sun) "movingLeft" true)))

  (on (clicked (prev sun)) (= sun (if (.. (prev sun) movingLeft) then (moveLeft (prev sun)) else (moveRight (prev sun)))))
)
"""


def generate_sexp(params: VariationParams) -> str:
    ml = "true" if params.sun_moving_left else "false"
    sun_init = f"(Sun {ml} (Position {params.sun_x} 0))"
    cloud_init = f"(Cloud (Position {params.cloud_x} 0))"

    leaf_parts = []
    for i, lx in enumerate(params.leaf_xs):
        h = params.leaf_heights_init[i]
        for dy in range(h):
            y = 15 - dy
            color = "mediumpurple" if y == 12 else "green"
            leaf_parts.append(f'(Leaf "{color}" (Position {lx} {y}))')
    leaves_init = "(list " + " ".join(leaf_parts) + ")"

    return SEXP_TEMPLATE.format(
        sun_init=sun_init,
        cloud_init=cloud_init,
        leaves_init=leaves_init,
    )


# ---------------------------------------------------------------------------
# Goal grid and mask construction
# ---------------------------------------------------------------------------
def build_goal_and_mask(params: VariationParams) -> Tuple[List[List[int]], List[List[int]]]:
    goal = [[COLOR_INT["black"]] * GRID_SIZE for _ in range(GRID_SIZE)]
    mask = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

    # Sun (3x3 gold) at initial position
    for dy in range(3):
        for dx in range(3):
            sx = params.sun_x + dx
            sy = dy
            if 0 <= sx < GRID_SIZE and 0 <= sy < GRID_SIZE:
                goal[sy][sx] = COLOR_INT["gold"]

    # Cloud at initial position
    cloud_offsets = [
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
    ]
    for dx, dy in cloud_offsets:
        cx = params.cloud_x + dx
        cy = dy
        if 0 <= cx < GRID_SIZE and 0 <= cy < GRID_SIZE:
            goal[cy][cx] = COLOR_INT["gray"]

    # Leaves at TARGET heights
    for i, lx in enumerate(params.leaf_xs):
        h_target = params.leaf_heights_target[i]
        for dy in range(h_target):
            y = 15 - dy
            color_int = COLOR_INT["mediumpurple"] if y == 12 else COLOR_INT["green"]
            goal[y][lx] = color_int

    # Mask: cover selected changed leaf columns (always full 4-row height, rows 12-15)
    for idx in params.mask_leaf_indices:
        lx = params.leaf_xs[idx]
        for y in range(12, 16):
            mask[y][lx] = 1

    return goal, mask


# ---------------------------------------------------------------------------
# Sun-cloud overlap detection
# ---------------------------------------------------------------------------
def sun_overlaps_cloud(sun_x: int, cloud_x: int) -> bool:
    """Check if sun [sun_x, sun_x+2] overlaps cloud [cloud_x-1, cloud_x+2]."""
    return sun_x <= cloud_x + 2 and sun_x + 2 >= cloud_x - 1


# ---------------------------------------------------------------------------
# Interpreter helpers
# ---------------------------------------------------------------------------
def apply_action(interpreter: "Interpreter", action: str) -> None:
    """Apply an action string to the interpreter and step."""
    if action == "left":
        interpreter.left()
    elif action == "right":
        interpreter.right()
    elif action == "down":
        interpreter.down()
    elif action == "up":
        interpreter.up()
    elif action.startswith("click"):
        parts = action.split()
        interpreter.click(int(parts[1]), int(parts[2]))
    # noop: no input method needed
    interpreter.step()


def render_to_color_grid(interpreter: "Interpreter") -> List[List[str]]:
    """Render current interpreter state to a 2D color-string grid."""
    scene_graph = json.loads(interpreter.render_all())
    bg = interpreter.get_background()
    grid_size = scene_graph.get("GRID_SIZE", GRID_SIZE)
    matrix = [[bg] * grid_size for _ in range(grid_size)]
    for key, objects in scene_graph.items():
        if key == "GRID_SIZE":
            continue
        for obj in objects:
            x = obj["position"]["x"]
            y = obj["position"]["y"]
            color = obj.get("color", key).lower()
            if 0 <= y < grid_size and 0 <= x < grid_size:
                matrix[y][x] = color
    return matrix


# ---------------------------------------------------------------------------
# Policy: compute and execute actions to reach target from initial state
# ---------------------------------------------------------------------------
def compute_and_verify(params: VariationParams, sexp_code: str,
                       goal_int: List[List[int]], mask: List[List[int]],
                       color_dict: Dict[int, str]) -> Tuple[bool, List[str]]:
    """Load sexp into Interpreter, run policy, verify goal reached.

    Returns (success, action_list).
    """
    interpreter = Interpreter()
    interpreter.run_script(sexp_code, autumnstdlib, "", 0)

    actions: List[str] = []

    # Local sun tracker
    sun_x = params.sun_x
    sun_ml = params.sun_moving_left
    # Boundary fixup for initial state (matches interpreter behavior on first step)
    if sun_x == 0:
        sun_ml = False
    if sun_x == 13:
        sun_ml = True

    cloud_x = params.cloud_x

    # Build growth tasks: (leaf_index, leaf_x, growths_needed)
    tasks = []
    for i, lx in enumerate(params.leaf_xs):
        needed = params.leaf_heights_target[i] - params.leaf_heights_init[i]
        if needed > 0:
            tasks.append((i, lx, needed))

    # Sort tasks by proximity to current cloud position (greedy nearest neighbor)
    ordered_tasks = []
    remaining = list(tasks)
    cur_cx = cloud_x
    while remaining:
        remaining.sort(key=lambda t: abs(t[1] - cur_cx))
        task = remaining.pop(0)
        ordered_tasks.append(task)
        cur_cx = task[1]

    for leaf_idx, leaf_x, growths_needed in ordered_tasks:
        # 1. Move cloud to leaf_x
        while cloud_x != leaf_x:
            if cloud_x < leaf_x:
                act = "right"
                cloud_x += 1
            else:
                act = "left"
                cloud_x -= 1
            apply_action(interpreter, act)
            actions.append(act)
            # Sun boundary updates happen each step (but sun doesn't move without click)
            _update_sun_no_click(sun_x, sun_ml)

        # 2. Ensure sun doesn't overlap cloud
        move_actions, sun_x, sun_ml = _move_sun_away(
            interpreter, sun_x, sun_ml, cloud_x, actions
        )
        if move_actions is None:
            return False, actions

        # 3. Grow leaf growths_needed times
        current_height = params.leaf_heights_init[leaf_idx]
        for g in range(growths_needed):
            top_y = 16 - current_height  # current topmost leaf y
            # Re-check sun overlap (in case cloud/sun config changed)
            if sun_overlaps_cloud(sun_x, cloud_x):
                mv, sun_x, sun_ml = _move_sun_away(
                    interpreter, sun_x, sun_ml, cloud_x, actions
                )
                if mv is None:
                    return False, actions

            # Press down to spawn water
            apply_action(interpreter, "down")
            actions.append("down")

            # Wait for water to fall and trigger growth
            # Water spawns at y=1, needs to reach y=top_y-1 as prev_water
            # then next step triggers growth. Total noop steps = top_y - 1.
            noop_count = top_y - 1
            for _ in range(noop_count):
                apply_action(interpreter, "noop")
                actions.append("noop")

            current_height += 1

    # Verify goal reached
    current_grid = render_to_color_grid(interpreter)
    goal_grid = [[color_dict[cell] for cell in row] for row in goal_int]
    success = check_grid_same(current_grid, goal_grid, mask)
    return success, actions


def _update_sun_no_click(sun_x: int, sun_ml: bool) -> Tuple[int, bool]:
    """Update sun tracker for a step without clicking."""
    if sun_x == 0:
        sun_ml = False
    elif sun_x == 13:
        sun_ml = True
    return sun_x, sun_ml


def _move_sun_away(
    interpreter: "Interpreter",
    sun_x: int,
    sun_ml: bool,
    cloud_x: int,
    actions: List[str],
) -> Tuple[Optional[List[str]], int, bool]:
    """Click the sun until it no longer overlaps the cloud.

    Returns (move_actions, new_sun_x, new_sun_ml) or (None, ...) on failure.
    """
    move_actions = []
    safety = 0
    while sun_overlaps_cloud(sun_x, cloud_x):
        safety += 1
        if safety > 30:
            return None, sun_x, sun_ml

        # Boundary edge case: wait a step for direction to flip
        if (sun_x == 0 and sun_ml) or (sun_x == 13 and not sun_ml):
            apply_action(interpreter, "noop")
            actions.append("noop")
            move_actions.append("noop")
            sun_x, sun_ml = _update_sun_no_click(sun_x, sun_ml)
            continue

        # Click the sun to move it
        click_x = sun_x + 1  # center-ish of 3x3 sun
        click_y = 1
        act = f"click {click_x} {click_y}"
        apply_action(interpreter, act)
        actions.append(act)
        move_actions.append(act)

        # After click: sun moves, movingLeft stays as prev value
        if sun_ml:
            sun_x -= 1
        else:
            sun_x += 1
        # sun_ml unchanged after click (click uses prev value)

    return move_actions, sun_x, sun_ml


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------
def sample_variation(rng: random.Random) -> Optional[VariationParams]:
    """Sample one random variation. Returns None if invalid."""
    num_leaves = rng.randint(4, 16)
    leaf_xs = sorted(rng.sample(range(16), num_leaves))

    leaf_heights_init = [rng.randint(1, 4) for _ in range(num_leaves)]
    leaf_heights_target = [rng.randint(h, 4) for h in leaf_heights_init]

    # Ensure at least one leaf actually grows
    changed = [i for i in range(num_leaves)
               if leaf_heights_target[i] > leaf_heights_init[i]]
    if not changed:
        growable = [i for i, h in enumerate(leaf_heights_init) if h < 4]
        if not growable:
            return None  # all maxed, skip
        idx = rng.choice(growable)
        leaf_heights_target[idx] = rng.randint(leaf_heights_init[idx] + 1, 4)
        changed = [idx]

    # Sun and cloud
    sun_x = rng.randint(0, 13)
    cloud_x = rng.randint(1, 14)
    sun_moving_left = rng.choice([True, False])
    if sun_x == 0:
        sun_moving_left = False
    if sun_x == 13:
        sun_moving_left = True

    # Mask: select 1..len(changed) of the changed leaves
    num_masked = rng.randint(1, len(changed))
    mask_leaf_indices = sorted(rng.sample(changed, num_masked))

    return VariationParams(
        sun_x=sun_x,
        cloud_x=cloud_x,
        sun_moving_left=sun_moving_left,
        leaf_xs=leaf_xs,
        leaf_heights_init=leaf_heights_init,
        leaf_heights_target=leaf_heights_target,
        mask_leaf_indices=mask_leaf_indices,
    )


def sample_variations(n: int, seed: int = 42) -> List[VariationParams]:
    rng = random.Random(seed)
    seen_hashes = set()
    variations = []
    attempts = 0
    max_attempts = n * 10

    while len(variations) < n and attempts < max_attempts:
        attempts += 1
        params = sample_variation(rng)
        if params is None:
            continue
        h = params.param_hash()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        variations.append(params)

    return variations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    print(f"Generating {n} variations of 7XF97 (seed={seed})...")

    # Load color dict
    color_dict = load_color_dict(EXAMPLE_BENCHMARK_DIR)

    # Prepare output dirs
    programs_dir = OUTPUT_DIR / "programs"
    prompts_dir = OUTPUT_DIR / "prompts"
    programs_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Copy color_dict.yaml
    shutil.copy2(COLOR_DICT_PATH, OUTPUT_DIR / "color_dict.yaml")

    # Sample variations
    variations = sample_variations(n, seed)
    print(f"Sampled {len(variations)} unique parameter sets.")

    # Generate, verify, write
    manifest_entries = []
    success_count = 0
    fail_count = 0

    for i, params in enumerate(variations):
        var_id = f"7XF97_v{i:04d}"
        sexp_code = generate_sexp(params)
        goal, mask = build_goal_and_mask(params)

        ok, actions = compute_and_verify(params, sexp_code, goal, mask, color_dict)

        if not ok:
            fail_count += 1
            print(f"  [{i:4d}] {var_id} FAILED verification "
                  f"(leaves={len(params.leaf_xs)}, actions={len(actions)})")
            continue

        success_count += 1

        # Write .sexp
        sexp_path = programs_dir / f"{var_id}.sexp"
        sexp_path.write_text(sexp_code, encoding="utf-8")

        # Write planning JSON
        planning_data = {
            "type": "planning",
            "program": var_id,
            "goal": goal,
            "mask": mask,
        }
        json_path = prompts_dir / f"{var_id}_planning.json"
        json_path.write_text(
            json.dumps(planning_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        manifest_entries.append({
            "id": var_id,
            "type": "planning",
            "program": var_id,
            "optimal_actions_count": len(actions),
            "params": {
                "sun_x": params.sun_x,
                "cloud_x": params.cloud_x,
                "sun_moving_left": params.sun_moving_left,
                "leaf_xs": params.leaf_xs,
                "leaf_heights_init": params.leaf_heights_init,
                "leaf_heights_target": params.leaf_heights_target,
                "mask_leaf_indices": params.mask_leaf_indices,
            },
        })

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(variations)} "
                  f"({success_count} ok, {fail_count} fail)")

    # Write manifest
    manifest = {
        "category": "generated_7XF97_variations",
        "base_program": "7XF97",
        "total_count": success_count,
        "files": manifest_entries,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone! {success_count} verified, {fail_count} failed.")
    print(f"Output: {OUTPUT_DIR}")

    if manifest_entries:
        action_counts = [e["optimal_actions_count"] for e in manifest_entries]
        print(f"Action counts: min={min(action_counts)}, "
              f"max={max(action_counts)}, "
              f"avg={sum(action_counts)/len(action_counts):.1f}")


if __name__ == "__main__":
    main()
