import argparse
import base64
import io
import json
import os
import sys
from typing import Iterable, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MARA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if MARA_ROOT not in sys.path:
    sys.path.insert(0, MARA_ROOT)

from generated.mara import mara_environment_pb2 as env_pb2  # noqa: E402
from python_examples.autumnbench.concrete_envs import PlanningEnvironment  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None


def _load_actions(actions_path: str) -> List[str]:
    with open(actions_path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            return []
        if isinstance(data[0], str):
            return data
        if isinstance(data[0], dict):
            actions: List[str] = []
            for item in data:
                if "action" in item:
                    actions.append(item["action"])
                elif "text_data" in item:
                    actions.append(item["text_data"])
                else:
                    raise ValueError(
                        "Unsupported action item format. Expected keys 'action' or 'text_data'."
                    )
            return actions
    raise ValueError("Unsupported actions.json format. Expected a list.")


def _find_actions_files(root_dir: str) -> List[str]:
    actions_files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "actions.json" not in filenames:
            continue
        if os.path.basename(dirpath) != "planning":
            continue
        actions_files.append(os.path.join(dirpath, "actions.json"))
    return sorted(actions_files)


def _overlay_frame_number(image_bytes: bytes, frame_idx: int) -> bytes:
    if Image is None:
        return image_bytes
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGBA")
        draw = ImageDraw.Draw(img)
        text = f"#{frame_idx:04d}"
        try:
            font = ImageFont.load_default()
        except Exception:  # pragma: no cover
            font = None
        text_bbox = draw.textbbox((0, 0), text, font=font)
        padding = 4
        x0, y0, x1, y1 = text_bbox
        box = [
            padding,
            padding,
            padding + (x1 - x0) + 6,
            padding + (y1 - y0) + 4,
        ]
        draw.rectangle(box, fill=(0, 0, 0, 160))
        draw.text((padding + 3, padding + 2), text, fill=(255, 255, 255, 255), font=font)
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()


def _save_image_data(image_data: bytes, output_dir: str, frame_idx: int) -> None:
    if not image_data:
        return
    payload = json.loads(image_data.decode("utf-8"))
    grid_b64 = payload.get("grid")
    goal_b64 = payload.get("goal_state")

    if grid_b64:
        grid_bytes = base64.b64decode(grid_b64)
        grid_bytes = _overlay_frame_number(grid_bytes, frame_idx)
        grid_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        with open(grid_path, "wb") as f:
            f.write(grid_bytes)

    goal_path = os.path.join(output_dir, "goal_state.png")
    if goal_b64 and not os.path.exists(goal_path):
        goal_bytes = base64.b64decode(goal_b64)
        with open(goal_path, "wb") as f:
            f.write(goal_bytes)


def _make_gif(frames: Iterable[str], output_path: str, fps: float) -> None:
    if Image is None:
        print("Pillow not available; skipping GIF generation.")
        return
    frames = list(frames)
    if not frames:
        return
    duration_ms = int(1000 / fps)
    images = [Image.open(frame).convert("RGBA") for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _infer_env_name(actions_path: str) -> Optional[str]:
    # Expected path: .../logs/<ENV>/planning/actions.json
    planning_dir = os.path.dirname(actions_path)
    env_dir = os.path.dirname(planning_dir)
    env_name = os.path.basename(env_dir)
    return env_name or None


def _replay_one(
    actions_path: str,
    env_name: str,
    data_dir: str,
    output_dir: str,
    fps: float,
    make_gif: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    actions = _load_actions(actions_path)

    env = PlanningEnvironment(
        env_name,
        render_mode="image",
        data_dir=data_dir,
        logging_path=output_dir,
    )

    frame_idx = 0
    initial_obs = env.get_observation()
    _save_image_data(initial_obs.image_data, output_dir, frame_idx)

    manifest = {
        "env": env_name,
        "data_dir": os.path.abspath(data_dir),
        "actions": actions,
        "frames": [],
    }
    if os.path.exists(os.path.join(output_dir, f"frame_{frame_idx:04d}.png")):
        manifest["frames"].append(f"frame_{frame_idx:04d}.png")

    for action in actions:
        frame_idx += 1
        obs, _, done, _ = env.step(env_pb2.Action(text_data=action))
        if obs.image_data:
            _save_image_data(obs.image_data, output_dir, frame_idx)
            frame_name = f"frame_{frame_idx:04d}.png"
            if os.path.exists(os.path.join(output_dir, frame_name)):
                manifest["frames"].append(frame_name)
        else:
            print(
                f"Warning: no image_data at step {frame_idx} for action '{action}'."
            )
            break
        if done:
            break

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if make_gif:
        frame_paths = [os.path.join(output_dir, name) for name in manifest["frames"]]
        gif_path = os.path.join(output_dir, "trajectory.gif")
        _make_gif(frame_paths, gif_path, fps=fps)

    print(f"Saved {len(manifest['frames'])} frames to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay planning actions and save trajectory frames."
    )
    parser.add_argument(
        "--actions",
        help="Path to actions.json (list of action strings).",
    )
    parser.add_argument(
        "--env",
        help="Environment name (program id), e.g. space_invaders."
             " If omitted, inferred from .../logs/<ENV>/planning/actions.json",
    )
    parser.add_argument(
        "--root",
        help=(
            "Root directory like experiments/full_evaluation/DAY/TIME. "
            "Will auto-visualize all */planning/actions.json under it."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(SCRIPT_DIR, "example_benchmark"),
        help="Data directory containing programs/, prompts/, color_dict.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for frames. Defaults to <actions_dir>/videos/.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="FPS for GIF output (default: 2).",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Disable GIF generation.",
    )

    args = parser.parse_args()

    make_gif = not args.no_gif

    if args.root:
        root_dir = os.path.abspath(args.root)
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Root directory not found: {root_dir}")
        actions_files = _find_actions_files(root_dir)
        if not actions_files:
            raise FileNotFoundError(
                f"No planning/actions.json found under: {root_dir}"
            )
        for actions_path in actions_files:
            env_name = args.env or _infer_env_name(actions_path)
            if not env_name:
                print(
                    f"Skipping {actions_path}: unable to infer env name."
                )
                continue
            if args.output_dir:
                output_dir = os.path.abspath(args.output_dir)
            else:
                output_dir = os.path.join(os.path.dirname(actions_path), "videos")
            _replay_one(
                actions_path=actions_path,
                env_name=env_name,
                data_dir=args.data_dir,
                output_dir=output_dir,
                fps=args.fps,
                make_gif=make_gif,
            )
        return

    if not args.actions:
        raise ValueError("Either --actions or --root must be provided.")

    actions_path = os.path.abspath(args.actions)
    if not os.path.exists(actions_path):
        raise FileNotFoundError(f"actions.json not found: {actions_path}")

    env_name = args.env or _infer_env_name(actions_path)
    if not env_name:
        raise ValueError("--env is required when env name cannot be inferred.")

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(actions_path), "videos")

    _replay_one(
        actions_path=actions_path,
        env_name=env_name,
        data_dir=args.data_dir,
        output_dir=output_dir,
        fps=args.fps,
        make_gif=make_gif,
    )


if __name__ == "__main__":
    main()
