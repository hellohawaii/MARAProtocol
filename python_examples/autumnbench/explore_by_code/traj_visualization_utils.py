from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .log_utils import allocate_timeline_index, run_logs_dir, safe_name, timeline_copy_file
except ImportError:  # pragma: no cover
    from log_utils import allocate_timeline_index, run_logs_dir, safe_name, timeline_copy_file

try:
    from PIL import Image, ImageColor, ImageDraw
except ImportError:  # pragma: no cover
    Image = None
    ImageColor = None
    ImageDraw = None


def _next_prefixed_index(parent_dir: Path) -> int:
    if not parent_dir.exists():
        return 1
    max_idx = 0
    for item in parent_dir.iterdir():
        name = item.name
        prefix, sep, _rest = name.partition("_")
        if not sep:
            continue
        try:
            num = int(prefix)
        except ValueError:
            continue
        if num > max_idx:
            max_idx = num
    return max_idx + 1


def _state_to_grid(state: Dict[str, Any]) -> Optional[List[List[str]]]:
    grid_size = state.get("GRID_SIZE")
    if not isinstance(grid_size, int) or grid_size <= 0:
        return None

    grid = [["black" for _ in range(grid_size)] for _ in range(grid_size)]
    object_types = sorted(state.keys())
    for obj_type in object_types:
        if obj_type == "GRID_SIZE":
            continue
        objs = state.get(obj_type)
        if not isinstance(objs, list):
            continue
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            pos = obj.get("position")
            color = obj.get("color")
            if not isinstance(pos, dict) or not isinstance(color, str):
                continue
            x, y = pos.get("x"), pos.get("y")
            if isinstance(x, int) and isinstance(y, int) and 0 <= x < grid_size and 0 <= y < grid_size:
                grid[y][x] = color
    return grid


def _color_to_rgb(color: str) -> tuple[int, int, int]:
    if ImageColor is None:
        return (0, 0, 0)
    try:
        return ImageColor.getrgb(color)
    except Exception:
        return (0, 0, 0)


def _render_grid_frame(state: Dict[str, Any], output_path: Path, cell_size: int = 20) -> bool:
    if Image is None or ImageDraw is None:
        return False
    grid = _state_to_grid(state)
    if not grid:
        return False

    grid_h = len(grid)
    grid_w = len(grid[0]) if grid_h else 0
    if grid_w <= 0:
        return False

    img = Image.new("RGB", (grid_w * cell_size, grid_h * cell_size), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y, row in enumerate(grid):
        for x, color_name in enumerate(row):
            x0 = x * cell_size
            y0 = y * cell_size
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1
            draw.rectangle([x0, y0, x1, y1], fill=_color_to_rgb(color_name))
            draw.rectangle([x0, y0, x1, y1], outline=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return True


def save_trajectory_visualization(
    payload: Dict[str, Any],
    out_path: Path,
    run_id: Optional[str],
) -> Dict[str, Any]:
    run_dir = run_logs_dir(run_id)
    traj_stem = safe_name(out_path.stem)
    traj_root = run_dir / "traj"
    traj_root.mkdir(parents=True, exist_ok=True)
    seq = _next_prefixed_index(traj_root)
    viz_dir = traj_root / f"{seq}_{traj_stem}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    transitions = payload.get("trajectory", [])
    if not isinstance(transitions, list) or not transitions:
        return {
            "frames_dir": str(viz_dir),
            "num_frames": 0,
            "gif_path": None,
            "gif_created": False,
            "reason": "empty trajectory",
        }

    states: List[Dict[str, Any]] = []
    first_state = transitions[0].get("state") if isinstance(transitions[0], dict) else None
    if isinstance(first_state, dict):
        states.append(first_state)
    for item in transitions:
        if isinstance(item, dict) and isinstance(item.get("new_state"), dict):
            states.append(item["new_state"])

    frame_paths: List[Path] = []
    for idx, state in enumerate(states):
        frame_path = viz_dir / f"frame_{idx:04d}.png"
        if _render_grid_frame(state, frame_path):
            frame_paths.append(frame_path)

    gif_path: Optional[Path] = None
    gif_created = False
    if Image is not None and frame_paths:
        images = []
        for path in frame_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGBA"))
        if images:
            gif_path = viz_dir / "trajectory.gif"
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=400,
                loop=0,
                optimize=False,
            )
            gif_created = True
            timeline_idx = allocate_timeline_index(run_id)
            timeline_copy_file(
                run_id=run_id,
                index=timeline_idx,
                stem=traj_stem,
                source_path=gif_path,
                suffix=".gif",
            )

    return {
        "frames_dir": str(viz_dir),
        "num_frames": len(frame_paths),
        "gif_path": str(gif_path) if gif_path else None,
        "gif_created": gif_created,
    }
