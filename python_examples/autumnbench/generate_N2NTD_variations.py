#!/usr/bin/env python3
"""Generate 1000 variations of the N2NTD Mario environment with different initial states.

Each variation has identical transition functions but different:
- Mario x-position
- Step positions (3-cell platforms)
- Coin positions
- Enemy x-position and direction

The goal is constant: kill the enemy (rows 0-1 all white).
Verification is done by loading each .sexp into the real Interpreter and running a policy.
"""

import copy
import hashlib
import json
import os
import random
import shutil
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "explore_by_code"))

from interpreter_module import Interpreter
from autumnstdlib import autumnstdlib
from planning_utils import load_color_dict, check_grid_same

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE = 12
EXAMPLE_BENCHMARK_DIR = SCRIPT_DIR / "example_benchmark"
OUTPUT_DIR = EXAMPLE_BENCHMARK_DIR / "program_with_testcases" / "N2NTD"
COLOR_DICT_PATH = EXAMPLE_BENCHMARK_DIR / "color_dict.yaml"

COLOR_INT = {
    "white": 8,
    "red": 11,
    "darkorange": 39,
    "gold": 4,
    "blue": 10,
    "mediumpurple": 6,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class VariationParams:
    mario_x: int
    steps: List[Tuple[int, int]]  # list of (origin_x, origin_y)
    coins: List[Tuple[int, int]]  # list of (x, y)
    enemy_x: int
    enemy_moving_left: bool

    def param_hash(self) -> str:
        payload = repr((
            self.mario_x, self.steps, self.coins,
            self.enemy_x, self.enemy_moving_left,
        ))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# .sexp generation
# ---------------------------------------------------------------------------
SEXP_TEMPLATE = """\
(program
  (= GRID_SIZE 12)
  (= background "white")
  (= FRAME_RATE 7)

  (object Mario (: bullets Number) (Cell 0 0 "red"))
  (object Step (: movingLeft Bool) (list (Cell -1 0 "darkorange") (Cell 0 0 "darkorange") (Cell 1 0 "darkorange")))
  (object Coin (Cell 0 0 "gold"))
  (object Enemy (: movingLeft Bool) (: lives Number) (list (Cell -1 0 "blue") (Cell 0 0 "blue") (Cell 1 0 "blue")
                                      (Cell -1 1 "blue") (Cell 0 1 "blue") (Cell 1 1 "blue")))
  (object Bullet (Cell 0 0 "mediumpurple"))

  (: mario Mario)
  (= mario (initnext {mario_init} (if (intersects (moveDown (prev mario)) (prev coins)) then (moveDown (prev mario)) else (moveDownNoCollision (prev mario)))))

  (: steps (List Step))
  (= steps (initnext {steps_init} (updateObj (prev steps) (--> step (if (.. step movingLeft) then (moveLeft step) else (moveRight step))))))

  (: coins (List Coin))
  (= coins (initnext {coins_init} (prev coins)))

  (: enemy Enemy)
  (= enemy (initnext {enemy_init} (if (.. (prev enemy) movingLeft) then (moveLeft (prev enemy)) else (moveRight (prev enemy)))))

  (: bullets (List Bullet))
  (= bullets (initnext (list) (updateObj (prev bullets) (--> obj (if (intersects (moveUp obj) (prev steps)) then (removeObj obj) else (moveUp obj))))))

  (: enemyLives Number)
  (= enemyLives (initnext 1 (prev enemyLives)))

  (on (& (defined "enemy") ( == (.. (.. (prev enemy) origin) x) 1)) (= enemy (moveRight (updateObj (prev enemy) "movingLeft" false))))
  (on (& (defined "enemy") (== (.. (.. (prev enemy) origin) x) 10)) (= enemy (moveLeft (updateObj (prev enemy) "movingLeft" true))))

  (on left (= mario (if (intersects (moveLeft (prev mario)) (prev coins)) then (moveLeft (prev mario)) else (moveLeftNoCollision (prev mario)))))
  (on right (= mario (if (intersects (moveRight (prev mario)) (prev coins)) then (moveRight (prev mario)) else (moveRightNoCollision (prev mario)))))
  (on (& ((up)) (== (moveDownNoCollision (prev mario)) (prev mario))) (= mario (moveNoCollision (prev mario) 0 -4)))

  (on true (= steps (updateObj (prev steps) (--> step (updateObj step "movingLeft" false)) (--> step (== (.. (.. step origin) x) 1)))))
  (on true (= steps (updateObj (prev steps) (--> step (updateObj step "movingLeft" true)) (--> step (== (.. (.. step origin) x) 10)))))

  (on (intersects (prev mario) (prev coins))
    (let (= coins (removeObj (prev coins) (--> (obj) (intersects obj (prev mario)))))
          (= mario (moveDownNoCollision (updateObj (prev mario) "bullets" (+ (.. (prev mario) bullets) 1))))) )

  (on (& ((clicked)) (> (.. (prev mario) bullets) 0))
    (let (= bullets (addObj (prev bullets) (Bullet (.. (prev mario) origin))))
          (= mario (moveDownNoCollision (updateObj (prev mario) "bullets" (- (.. (prev mario) bullets) 1))))
          true
          ))

  (on (& (defined "enemy") (intersects (prev enemy) (prev bullets)))
    (let (= bullets (removeObj (prev bullets) (--> obj (intersects obj (prev enemy)))))
          (= enemy (if (== (prev enemyLives) 1) then (removeObj (prev enemy)) else (if (.. (prev enemy) movingLeft) then (moveLeft (prev enemy)) else (moveRight (prev enemy))) ))
          (= enemyLives (- (prev enemyLives) 1)))
          true
    )
)
"""


def generate_sexp(params: VariationParams) -> str:
    mario_init = f"(Mario 0 (Position {params.mario_x} 11))"

    step_parts = []
    for sx, sy in params.steps:
        step_parts.append(f"(Step true (Position {sx} {sy}))")
    steps_init = "(list " + " ".join(step_parts) + ")" if step_parts else "(list)"

    coin_parts = []
    for cx, cy in params.coins:
        coin_parts.append(f"(Coin (Position {cx} {cy}))")
    coins_init = "(list " + " ".join(coin_parts) + ")" if coin_parts else "(list)"

    ml = "true" if params.enemy_moving_left else "false"
    enemy_init = f"(Enemy {ml} 1 (Position {params.enemy_x} 0))"

    return SEXP_TEMPLATE.format(
        mario_init=mario_init,
        steps_init=steps_init,
        coins_init=coins_init,
        enemy_init=enemy_init,
    )


# ---------------------------------------------------------------------------
# Goal grid and mask construction (constant)
# ---------------------------------------------------------------------------
def build_goal_and_mask() -> Tuple[List[List[int]], List[List[int]]]:
    bg = COLOR_INT["white"]
    goal = [[bg] * GRID_SIZE for _ in range(GRID_SIZE)]
    mask = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    # Mask rows 0-1: enemy should be dead (all white)
    for y in range(2):
        for x in range(GRID_SIZE):
            mask[y][x] = 1
    return goal, mask


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def compute_step_cells(steps: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    cells = set()
    for sx, sy in steps:
        for dx in [-1, 0, 1]:
            cx = sx + dx
            if 0 <= cx < GRID_SIZE:
                cells.add((cx, sy))
    return cells


def compute_stable_set(step_cells: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    stable = set()
    for x in range(GRID_SIZE):
        stable.add((x, GRID_SIZE - 1))  # ground y=11
    for sx, sy in step_cells:
        above = sy - 1
        if above >= 0 and (sx, above) not in step_cells:
            stable.add((sx, above))
    return stable


def get_walkable_xs(start_x: int, y: int, step_cells: Set[Tuple[int, int]]) -> List[int]:
    result = [start_x]
    for x in range(start_x + 1, GRID_SIZE):
        if (x, y) in step_cells:
            break
        result.append(x)
    for x in range(start_x - 1, -1, -1):
        if (x, y) in step_cells:
            break
        result.append(x)
    return result


def fall_from_y(x: int, start_y: int, step_cells: Set[Tuple[int, int]]) -> int:
    for y in range(start_y + 1, GRID_SIZE):
        if (x, y) in step_cells:
            return y - 1
    return GRID_SIZE - 1


# ---------------------------------------------------------------------------
# BFS reachability
# ---------------------------------------------------------------------------
def bfs_reachability(
    mario_x: int,
    step_cells: Set[Tuple[int, int]],
    coins: List[Tuple[int, int]],
    start_y: int = GRID_SIZE - 1,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Dict]:
    """BFS on stable positions.

    Returns (reachable_stable, accessible_coins, parent_dict).
    parent_dict maps pos -> (from_pos, transition_type, ...) or None for start.
    """
    stable = compute_stable_set(step_cells)
    coin_set = set(coins)

    start = (mario_x, start_y)
    reachable = set()
    parent: Dict[Tuple[int, int], Optional[tuple]] = {}
    accessible = set()

    if start in stable:
        reachable.add(start)
        parent[start] = None
        queue: deque = deque([start])
    else:
        return reachable, accessible, parent

    while queue:
        pos = queue.popleft()
        x, y = pos

        # Coins reachable by walking at height y
        walkable = get_walkable_xs(x, y, step_cells)
        for wx in walkable:
            if (wx, y) in coin_set:
                accessible.add((wx, y))

        # Jump check
        jy = y - 4
        can_jump = (jy >= 2
                    and (x, jy) not in step_cells
                    and (x, jy) not in coin_set)

        if can_jump:
            jump_walkable = get_walkable_xs(x, jy, step_cells)
            for jx in jump_walkable:
                # Coin at jump height
                if (jx, jy) in coin_set:
                    accessible.add((jx, jy))
                # Coins on fall path from (jx, jy)
                for fy in range(jy + 1, GRID_SIZE):
                    if (jx, fy) in step_cells:
                        break
                    if (jx, fy) in coin_set:
                        accessible.add((jx, fy))

        # Generate neighbor stable positions via walk
        for wx in walkable:
            wp = (wx, y)
            if wp in stable and wp not in reachable:
                reachable.add(wp)
                parent[wp] = (pos, "walk")
                queue.append(wp)

        # Walk-off-edge: walking to a non-stable position and falling
        if y < GRID_SIZE - 1:
            for wx in walkable:
                if (wx, y) not in stable:
                    landing_y = fall_from_y(wx, y, step_cells)
                    lp = (wx, landing_y)
                    if lp in stable and lp not in reachable:
                        reachable.add(lp)
                        parent[lp] = (pos, "walk_fall", wx, landing_y)
                        queue.append(lp)

        # Generate neighbors via jump+walk+fall
        if can_jump:
            jump_walkable = get_walkable_xs(x, jy, step_cells)
            for jx in jump_walkable:
                jp = (jx, jy)
                if jp in stable and jp not in reachable:
                    reachable.add(jp)
                    parent[jp] = (pos, "jump", x, jx, jy)
                    queue.append(jp)
                if jp not in stable:
                    landing_y = fall_from_y(jx, jy, step_cells)
                    lp = (jx, landing_y)
                    if lp in stable and lp not in reachable:
                        reachable.add(lp)
                        parent[lp] = (pos, "jump_fall", x, jx, jy)
                        queue.append(lp)

    return reachable, accessible, parent


# ---------------------------------------------------------------------------
# Shootable positions
# ---------------------------------------------------------------------------
def find_shootable_positions(
    reachable: Set[Tuple[int, int]],
    step_cells: Set[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Find reachable stable positions with clear column above to y=0."""
    shootable = []
    for sx, sy in reachable:
        clear = True
        for y in range(sy):
            if (sx, y) in step_cells:
                clear = False
                break
        if clear:
            shootable.append((sx, sy))
    return shootable


# ---------------------------------------------------------------------------
# Enemy simulation
# ---------------------------------------------------------------------------
def simulate_enemy_step(x: int, ml: bool) -> Tuple[int, bool]:
    if x == 1:
        return 2, False
    elif x == 10:
        return 9, True
    else:
        return (x - 1 if ml else x + 1), ml


def simulate_enemy(x: int, ml: bool, n_steps: int) -> Tuple[int, bool]:
    for _ in range(n_steps):
        x, ml = simulate_enemy_step(x, ml)
    return x, ml


# ---------------------------------------------------------------------------
# Interpreter helpers
# ---------------------------------------------------------------------------
def apply_action(interpreter: "Interpreter", action: str) -> None:
    if action == "left":
        interpreter.left()
    elif action == "right":
        interpreter.right()
    elif action == "up":
        interpreter.up()
    elif action == "down":
        interpreter.down()
    elif action.startswith("click"):
        parts = action.split()
        interpreter.click(int(parts[1]), int(parts[2]))
    interpreter.step()


def render_to_color_grid(interpreter: "Interpreter") -> List[List[str]]:
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


def has_enemy(interpreter: "Interpreter") -> bool:
    scene_graph = json.loads(interpreter.render_all())
    for key, objects in scene_graph.items():
        if key == "GRID_SIZE":
            continue
        for obj in objects:
            if obj.get("color", "").lower() == "blue":
                return True
    return False


# ---------------------------------------------------------------------------
# Path planning and policy execution
# ---------------------------------------------------------------------------
def reconstruct_path(
    parent: Dict, target: Tuple[int, int]
) -> List[Tuple[Tuple[int, int], Optional[tuple]]]:
    """Reconstruct path from start to target using parent dict.
    Returns list of (position, transition_info) from start to target.
    """
    path = []
    pos = target
    while pos is not None and parent.get(pos) is not None:
        info = parent[pos]
        path.append((pos, info))
        pos = info[0]  # parent position
    path.reverse()
    return path


def find_best_coin_target(
    mario_x: int,
    step_cells: Set[Tuple[int, int]],
    coins: List[Tuple[int, int]],
    reachable: Set[Tuple[int, int]],
    accessible: Set[Tuple[int, int]],
    parent: Dict,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    """Find the best coin to collect.

    Returns (coin_pos, parent_stable_pos, collect_type) or None.
    collect_type is "walk", "jump_walk", or "jump_walk_fall".
    """
    coin_set = set(coins)
    best = None
    best_cost = float("inf")

    for pos in reachable:
        x, y = pos
        # Cost = number of transitions in BFS path
        cost = 0
        p = pos
        while p is not None and parent.get(p) is not None:
            cost += 1
            p = parent[p][0]

        # Check coins reachable by walking from this position
        walkable = get_walkable_xs(x, y, step_cells)
        for wx in walkable:
            if (wx, y) in coin_set:
                walk_cost = cost + abs(wx - x)
                if walk_cost < best_cost:
                    best_cost = walk_cost
                    best = ((wx, y), pos, "walk")

        # Check coins reachable by jump from this position
        jy = y - 4
        if (jy >= 2
                and (x, jy) not in step_cells
                and (x, jy) not in coin_set):
            jump_walkable = get_walkable_xs(x, jy, step_cells)
            for jx in jump_walkable:
                if (jx, jy) in coin_set:
                    jcost = cost + 1 + abs(jx - x)
                    if jcost < best_cost:
                        best_cost = jcost
                        best = ((jx, jy), pos, "jump_walk")
                # Fall path coins
                for fy in range(jy + 1, GRID_SIZE):
                    if (jx, fy) in step_cells:
                        break
                    if (jx, fy) in coin_set:
                        fcost = cost + 1 + abs(jx - x) + (fy - jy)
                        if fcost < best_cost:
                            best_cost = fcost
                            best = ((jx, fy), pos, "jump_walk_fall")

    return best


def execute_walk(actions: List[str], from_x: int, to_x: int) -> int:
    """Append walk actions. Returns new x."""
    while from_x != to_x:
        if to_x > from_x:
            actions.append("right")
            from_x += 1
        else:
            actions.append("left")
            from_x -= 1
    return from_x


def execute_bfs_path(
    actions: List[str],
    cur_x: int,
    cur_y: int,
    path: List[Tuple[Tuple[int, int], Optional[tuple]]],
) -> Tuple[int, int]:
    """Execute a BFS path (walk / jump / jump_fall transitions).
    Appends to *actions* and returns (final_x, final_y).
    """
    for dest_pos, transition in path:
        ttype = transition[1]
        if ttype == "walk":
            cur_x = execute_walk(actions, cur_x, dest_pos[0])
            cur_y = dest_pos[1]
        elif ttype == "jump":
            jump_from_x = transition[2]
            walk_to_x = transition[3]
            jump_y = transition[4]
            cur_x = execute_walk(actions, cur_x, jump_from_x)
            actions.append("up")
            cur_y = jump_y
            cur_x = execute_walk(actions, cur_x, walk_to_x)
        elif ttype == "walk_fall":
            walk_to_x = transition[2]
            landing_y = transition[3]
            cur_x = execute_walk(actions, cur_x, walk_to_x)
            for _ in range(landing_y - cur_y):
                actions.append("noop")
            cur_y = landing_y
        elif ttype == "jump_fall":
            jump_from_x = transition[2]
            walk_to_x = transition[3]
            jump_y = transition[4]
            cur_x = execute_walk(actions, cur_x, jump_from_x)
            actions.append("up")
            cur_y = jump_y
            cur_x = execute_walk(actions, cur_x, walk_to_x)
            landing_y = dest_pos[1]
            for _ in range(landing_y - cur_y):
                actions.append("noop")
            cur_y = landing_y
    return cur_x, cur_y


def plan_actions_to_collect(
    mario_x: int,
    step_cells: Set[Tuple[int, int]],
    coins: List[Tuple[int, int]],
    reachable: Set[Tuple[int, int]],
    accessible: Set[Tuple[int, int]],
    parent: Dict,
) -> Optional[List[str]]:
    """Plan action sequence to collect a coin. Returns action list or None."""
    target = find_best_coin_target(
        mario_x, step_cells, coins, reachable, accessible, parent
    )
    if target is None:
        return None

    coin_pos, parent_stable, collect_type = target
    actions: List[str] = []

    # Navigate to parent_stable via BFS path
    path = reconstruct_path(parent, parent_stable)
    cur_x, cur_y = execute_bfs_path(actions, mario_x, GRID_SIZE - 1, path)

    # Collect the coin.
    cx, cy = coin_pos

    if collect_type == "walk":
        cur_x = execute_walk(actions, cur_x, cx)
        actions.append("noop")
        actions.append("noop")
    elif collect_type == "jump_walk":
        actions.append("up")
        cur_y -= 4
        cur_x = execute_walk(actions, cur_x, cx)
        actions.append("noop")
        actions.append("noop")
        for _ in range(8):
            actions.append("noop")
    elif collect_type == "jump_walk_fall":
        actions.append("up")
        cur_y -= 4
        cur_x = execute_walk(actions, cur_x, cx)
        fall_dist = cy - cur_y + 3
        for _ in range(fall_dist):
            actions.append("noop")

    return actions


def find_shoot_candidates(
    step_cells: Set[Tuple[int, int]],
    mario_x: int,
) -> List[Tuple[int, int]]:
    """Build ordered list of shoot position candidates.

    Priority:
    1. Open columns (no step cells anywhere) -> ground position (x, 11)
    2. Columns with steps -> on top of the highest step in the column

    Within each group, sorted by distance to mario_x.
    """
    candidates: List[Tuple[int, int]] = []

    # Step 1: Open columns -> ground level
    open_cols = []
    for x in range(GRID_SIZE):
        if not any((x, y) in step_cells for y in range(GRID_SIZE)):
            open_cols.append((x, GRID_SIZE - 1))
    open_cols.sort(key=lambda p: abs(p[0] - mario_x))
    candidates.extend(open_cols)

    # Step 2: Columns with steps -> top of highest step
    step_tops = []
    for x in range(GRID_SIZE):
        highest_y = None
        for y in range(GRID_SIZE):
            if (x, y) in step_cells:
                highest_y = y
                break
        if highest_y is None:
            continue
        shoot_y = highest_y - 1
        if shoot_y < 2:  # too close to enemy zone
            continue
        step_tops.append((x, shoot_y))
    step_tops.sort(key=lambda p: abs(p[0] - mario_x))
    candidates.extend(step_tops)

    return candidates


def find_shoot_timing(
    params: VariationParams,
    total_steps: int,
    shoot_x: int,
    shoot_y: int,
) -> Optional[int]:
    """Find wait time before shooting so bullet hits enemy."""
    for wait in range(30):
        # Bullet at y=1: check at step total_steps + wait + shoot_y
        enemy_step_y1 = total_steps + wait + shoot_y
        ex1, _ = simulate_enemy(params.enemy_x, params.enemy_moving_left, enemy_step_y1)
        if abs(shoot_x - ex1) <= 1:
            return wait
        # Bullet at y=0: check at step total_steps + wait + shoot_y + 1
        enemy_step_y0 = total_steps + wait + shoot_y + 1
        ex0, _ = simulate_enemy(params.enemy_x, params.enemy_moving_left, enemy_step_y0)
        if abs(shoot_x - ex0) <= 1:
            return wait
    return None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def compute_and_verify(
    params: VariationParams,
    sexp_code: str,
    goal_int: List[List[int]],
    mask: List[List[int]],
    color_dict: Dict[int, str],
) -> Tuple[bool, List[str], bool]:
    """Load sexp into Interpreter, run policy, verify goal reached.

    Returns (success, actions, coin_collected).
    """
    step_cells = compute_step_cells(params.steps)
    stable = compute_stable_set(step_cells)
    reachable, accessible, parent = bfs_reachability(
        params.mario_x, step_cells, params.coins
    )

    if not accessible:
        return False, [], False

    # Plan coin collection
    collect_actions = plan_actions_to_collect(
        params.mario_x, step_cells, params.coins,
        reachable, accessible, parent,
    )
    if collect_actions is None:
        return False, [], False

    # Load interpreter
    interpreter = Interpreter()
    interpreter.run_script(sexp_code, autumnstdlib, "", 0)

    all_actions: List[str] = []

    # Execute coin collection
    for action in collect_actions:
        apply_action(interpreter, action)
        all_actions.append(action)

    # Check if the targeted coin was collected (fewer coins than initially)
    scene_after_collect = json.loads(interpreter.render_all())
    coins_remaining = sum(
        1 for key, objects in scene_after_collect.items() if key != "GRID_SIZE"
        for obj in objects if obj.get("color", "").lower() == "gold"
    )
    coin_collected = coins_remaining < len(params.coins)

    # Check if enemy already dead (shouldn't be, but check)
    if not has_enemy(interpreter):
        current_grid = render_to_color_grid(interpreter)
        goal_grid = [[color_dict[cell] for cell in row] for row in goal_int]
        if check_grid_same(current_grid, goal_grid, mask):
            return True, all_actions, coin_collected
        return False, all_actions, coin_collected

    # Determine Mario's current position by rendering
    scene = json.loads(interpreter.render_all())
    mario_pos = None
    for key, objects in scene.items():
        if key == "GRID_SIZE":
            continue
        for obj in objects:
            if obj.get("color", "").lower() == "red":
                mario_pos = (obj["position"]["x"], obj["position"]["y"])
                break
        if mario_pos:
            break

    if mario_pos is None:
        return False, all_actions, coin_collected

    # --- Shoot phase: navigate to shoot position, verify, then fire ---
    # Retry loop handles cases where uncollected coins in the nav path
    # cause the on-intersects handler to override movement for a step,
    # shifting Mario's actual position from the expected one.
    success = False
    for attempt in range(3):
        # Read Mario's actual position from interpreter
        scene = json.loads(interpreter.render_all())
        mx, my = None, None
        for key, objects in scene.items():
            if key == "GRID_SIZE":
                continue
            for obj in objects:
                if obj.get("color", "").lower() == "red":
                    mx, my = obj["position"]["x"], obj["position"]["y"]
                    break
            if mx is not None:
                break
        if mx is None:
            break

        # Fall to stable position (gravity only — just noops)
        stable_set = compute_stable_set(step_cells)
        if (mx, my) not in stable_set:
            landing_y = fall_from_y(mx, my, step_cells)
            for _ in range(landing_y - my):
                apply_action(interpreter, "noop")
                all_actions.append("noop")
            my = landing_y

        # BFS from current stable position (may be on a step, not ground)
        _, _, nav_parent = bfs_reachability(mx, step_cells, [], start_y=my)

        # Build shoot candidates: open columns first, then step tops
        candidates = find_shoot_candidates(step_cells, mx)

        attempted = False
        for shoot_pos in candidates:
            # Check BFS reachability
            if shoot_pos != (mx, my) and shoot_pos not in nav_parent:
                continue

            # Plan navigation actions
            if shoot_pos == (mx, my):
                nav_actions: List[str] = []
            else:
                path = reconstruct_path(nav_parent, shoot_pos)
                nav_actions = []
                execute_bfs_path(nav_actions, mx, my, path)

            # Check shoot timing BEFORE executing navigation
            sx, sy = shoot_pos
            test_total = len(all_actions) + len(nav_actions)
            wait = find_shoot_timing(params, test_total, sx, sy)
            if wait is None:
                continue

            # Found a valid candidate — execute navigation
            for action in nav_actions:
                apply_action(interpreter, action)
                all_actions.append(action)
            attempted = True

            # Verify Mario reached the target position
            scene = json.loads(interpreter.render_all())
            actual_pos = None
            for key, objects in scene.items():
                if key == "GRID_SIZE":
                    continue
                for obj in objects:
                    if obj.get("color", "").lower() == "red":
                        actual_pos = (obj["position"]["x"], obj["position"]["y"])
                        break
                if actual_pos:
                    break

            if actual_pos != shoot_pos:
                break  # Position mismatch, retry from actual position

            # Execute wait + shoot + bullet travel
            for _ in range(wait):
                apply_action(interpreter, "noop")
                all_actions.append("noop")

            apply_action(interpreter, "click 0 0")
            all_actions.append("click 0 0")

            for _ in range(sy + 3):
                apply_action(interpreter, "noop")
                all_actions.append("noop")

            # Verify goal
            current_grid = render_to_color_grid(interpreter)
            goal_grid = [[color_dict[cell] for cell in row] for row in goal_int]
            success = check_grid_same(current_grid, goal_grid, mask)

            if not success:
                for _ in range(5):
                    apply_action(interpreter, "noop")
                    all_actions.append("noop")
                current_grid = render_to_color_grid(interpreter)
                success = check_grid_same(current_grid, goal_grid, mask)

            break  # Committed to this candidate (shot fired)

        if success or not attempted:
            break
        if attempted and actual_pos == shoot_pos:
            break  # Shot was fired but goal check failed

    return success, all_actions, coin_collected


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------
def _steps_overlap(steps: List[Tuple[int, int]]) -> bool:
    """Check if any step cells overlap."""
    cells = set()
    for sx, sy in steps:
        for dx in [-1, 0, 1]:
            cx = sx + dx
            pos = (cx, sy)
            if pos in cells:
                return True
            cells.add(pos)
    return False


def sample_variation(rng: random.Random) -> Optional[VariationParams]:
    """Sample one random variation. Returns None if invalid."""
    # Choose difficulty: 50/50 medium/hard
    difficulty = rng.choice(["medium", "hard"])

    if difficulty == "medium":
        num_steps = 3
        num_coins = 2
    else:
        num_steps = 5
        num_coins = 1

    # Sample step positions
    steps: List[Tuple[int, int]] = []
    for _ in range(num_steps * 10):  # retry loop
        sx = rng.randint(1, 10)
        sy = rng.randint(3, 10)
        candidate = steps + [(sx, sy)]
        if not _steps_overlap(candidate):
            # Check cells are in bounds
            if sx - 1 >= 0 and sx + 1 < GRID_SIZE:
                steps.append((sx, sy))
        if len(steps) == num_steps:
            break

    if len(steps) != num_steps:
        return None

    step_cells = compute_step_cells(steps)

    # Sample coin positions
    coins: List[Tuple[int, int]] = []
    for _ in range(num_coins * 20):
        cx = rng.randint(0, GRID_SIZE - 1)
        cy = rng.randint(2, GRID_SIZE - 1)  # not in enemy zone (rows 0-1)
        if (cx, cy) not in step_cells:
            # Don't place coin where it would block a useful jump
            if (cx, cy) not in coins:
                coins.append((cx, cy))
        if len(coins) == num_coins:
            break

    if len(coins) != num_coins:
        return None

    # Sample mario position
    mario_x = rng.randint(0, GRID_SIZE - 1)

    # Sample enemy
    enemy_x = rng.randint(1, 10)
    enemy_moving_left = rng.choice([True, False])
    if enemy_x == 1:
        enemy_moving_left = False
    if enemy_x == 10:
        enemy_moving_left = True

    params = VariationParams(
        mario_x=mario_x,
        steps=steps,
        coins=coins,
        enemy_x=enemy_x,
        enemy_moving_left=enemy_moving_left,
    )

    # Quick validation: check BFS reachability
    reachable, accessible, parent = bfs_reachability(mario_x, step_cells, coins)
    if not accessible:
        return None

    shootable = find_shootable_positions(reachable, step_cells)
    if not shootable:
        return None

    return params


def sample_variations(n: int, seed: int = 42) -> List[VariationParams]:
    rng = random.Random(seed)
    seen_hashes = set()
    variations = []
    attempts = 0
    max_attempts = n * 20

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

    print(f"Generating {n} variations of N2NTD (seed={seed})...")

    color_dict = load_color_dict(EXAMPLE_BENCHMARK_DIR)
    goal, mask = build_goal_and_mask()

    programs_dir = OUTPUT_DIR / "programs"
    prompts_dir = OUTPUT_DIR / "prompts"
    programs_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(COLOR_DICT_PATH, OUTPUT_DIR / "color_dict.yaml")

    variations = sample_variations(n, seed)
    print(f"Sampled {len(variations)} unique parameter sets.")

    manifest_entries = []
    success_count = 0
    fail_count = 0

    for i, params in tqdm(enumerate(variations), total=len(variations)) :
        var_id = f"N2NTD_v{i:04d}"
        sexp_code = generate_sexp(params)

        ok, actions, coin_collected = compute_and_verify(
            params, sexp_code, goal, mask, color_dict
        )

        if not ok:
            fail_count += 1
            reason = "coin" if not coin_collected else "shoot"
            print(f"  [{i:4d}] {var_id} FAILED ({reason}) "
                  f"(steps={len(params.steps)}, coins={len(params.coins)}, "
                  f"actions={len(actions)})")
        else:
            success_count += 1

        # Save program and prompt for all variations (verified or not)
        sexp_path = programs_dir / f"{var_id}.sexp"
        sexp_path.write_text(sexp_code, encoding="utf-8")

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
            "verified": ok,
            "coin_collected": coin_collected,
            "optimal_actions_count": len(actions) if ok else None,
            "params": {
                "mario_x": params.mario_x,
                "steps": params.steps,
                "coins": [list(c) for c in params.coins],
                "enemy_x": params.enemy_x,
                "enemy_moving_left": params.enemy_moving_left,
            },
        })

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(variations)} "
                  f"({success_count} ok, {fail_count} fail)")

    manifest = {
        "category": "generated_N2NTD_variations",
        "base_program": "N2NTD",
        "total_count": len(manifest_entries),
        "verified_count": success_count,
        "failed_count": fail_count,
        "files": manifest_entries,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save list of failed programs where coin collection is unreachable
    # (not shoot-timing misses — those are valid games, just unverified)
    failed_coin = [
        e["id"] for e in manifest_entries
        if not e["verified"] and not e.get("coin_collected", True)
    ]
    failed_path = OUTPUT_DIR / "failed_programs.json"
    failed_path.write_text(
        json.dumps(failed_coin, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone! {success_count} verified, {fail_count} failed.")
    print(f"Output: {OUTPUT_DIR}")
    if failed_coin:
        print(f"Coin-unreachable failures: {len(failed_coin)}")

    if manifest_entries:
        action_counts = [e["optimal_actions_count"] for e in manifest_entries
                         if e["optimal_actions_count"] is not None]
        print(f"Action counts: min={min(action_counts)}, "
              f"max={max(action_counts)}, "
              f"avg={sum(action_counts)/len(action_counts):.1f}")


if __name__ == "__main__":
    main()
