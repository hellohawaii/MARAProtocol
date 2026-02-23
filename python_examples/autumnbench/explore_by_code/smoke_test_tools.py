import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from env_wrapper import (
    execute_run_command,
    get_runtime_info,
    get_langchain_tools,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def run_isolation_probe(label: str) -> dict:
    script = (
        "import json\n"
        "from env_wrapper import execute_run_command, get_runtime_info\n"
        "execute_run_command(\"python -c \\\"print('probe ok')\\\"\")\n"
        f"execute_run_command(\"sh -lc \\\"echo {label} > isolation_marker.txt\\\"\")\n"
        "info = get_runtime_info()\n"
        f"print(json.dumps({{'label': '{label}', 'run_id': info.get('run_id'), 'workspace_dir': info.get('workspace_dir')}}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"isolation probe failed ({label}): {proc.stderr.strip()}")
    output = proc.stdout.strip().splitlines()
    if not output:
        raise RuntimeError(f"isolation probe produced no output ({label})")
    return json.loads(output[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for explore_by_code tools")
    parser.add_argument("--env-name", default="7XF97", help="Env name for optional API test")
    parser.add_argument("--test-env-api", action="store_true", help="Also test env.reset/step/save_trajectory through container API client")
    parser.add_argument("--obfuscated", action="store_true", help="Use obfuscated env and python_programs_obfuscated for trajectory check")
    args = parser.parse_args()

    print_header("1) execute_run_command")
    cmd_msg = execute_run_command("python -c \"print('run_command ok')\"")
    print(cmd_msg)

    print_header("2) Persistent container state check")
    execute_run_command("sh -lc \"echo persistent > _persist_check.txt\"")
    persist_msg = execute_run_command("cat _persist_check.txt")
    print(persist_msg)

    runtime_info = get_runtime_info()
    workspace = Path(runtime_info["workspace_dir"]) if runtime_info.get("workspace_dir") else (Path(__file__).resolve().parent / "llm_workspace")
    out_file = workspace / "_persist_check.txt"
    print(f"Persistent file exists: {out_file.exists()} -> {out_file}")

    print_header("3) LangChain tools")
    tools = get_langchain_tools()
    run_tool = next(t for t in tools if t.name == "run_command_in_docker")

    tool_run_result = run_tool.invoke({"command": "python -c \"print('tool run ok')\""})
    print("run_command_in_docker tool ->", tool_run_result)

    if args.test_env_api:
        print_header("4) Optional env API via container")
        env_test_script = (
            "python - <<'PY'\n"
            "import json\n"
            "from env_api_client import RemoteEnvWrapper\n"
            f"env = RemoteEnvWrapper()\n"
            f"r = env.reset(env_name='{args.env_name}', seed=0)\n"
            "s = env.step('noop')\n"
            "t = env.save_trajectory('smoke_traj')\n"
            "print(json.dumps({'reset_ok': r.get('ok'), 'step_ok': s.get('ok'), 'obfuscated': r.get('obfuscated'), 'traj_path': t.get('path')}, ensure_ascii=False))\n"
            "PY"
        )
        env_api_result = execute_run_command(env_test_script)
        print(env_api_result)

    print_header("5) Check trajectory with external tool")
    programs_dir = "python_programs_obfuscated" if args.obfuscated else "python_programs"
    source_program = Path(__file__).resolve().parent.parent / "example_benchmark" / programs_dir / f"{args.env_name}.py"
    target_program = workspace / f"test_{args.env_name}.py"
    if source_program.exists():
        shutil.copy(source_program, target_program)
        print(f"Copied {source_program.name} (from {programs_dir}) to {target_program.name}")
    else:
        print(f"Warning: {source_program} does not exist, check_traj_example might fail.")

    check_cmd = f"python check_traj_example.py test_{args.env_name}.py traj/{args.env_name}/"
    check_result = execute_run_command(check_cmd)
    print(check_result)

    print_header("6) Isolation check (two independent runs)")
    run1 = run_isolation_probe("run_one")
    run2 = run_isolation_probe("run_two")
    print("run1 ->", run1)
    print("run2 ->", run2)

    run1_workspace = Path(run1["workspace_dir"])
    run2_workspace = Path(run2["workspace_dir"])
    run1_marker = (run1_workspace / "isolation_marker.txt").read_text(encoding="utf-8").strip()
    run2_marker = (run2_workspace / "isolation_marker.txt").read_text(encoding="utf-8").strip()
    isolated = (
        run1["run_id"] != run2["run_id"]
        and run1_workspace != run2_workspace
        and run1_marker == "run_one"
        and run2_marker == "run_two"
    )
    print(
        json.dumps(
            {
                "isolated": isolated,
                "run1_workspace": str(run1_workspace),
                "run2_workspace": str(run2_workspace),
                "run1_marker": run1_marker,
                "run2_marker": run2_marker,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print_header("DONE")
    print("Smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
