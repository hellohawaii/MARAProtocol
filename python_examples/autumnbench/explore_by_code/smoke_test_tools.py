import argparse
import json
from pathlib import Path

from env_wrapper import (
    execute_run_command,
    execute_write_file,
    get_langchain_tools,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for explore_by_code tools")
    parser.add_argument("--env-name", default="7XF97", help="Env name for optional API test")
    parser.add_argument("--test-env-api", action="store_true", help="Also test env.reset/step/save_trajectory through container API client")
    args = parser.parse_args()

    print_header("1) execute_write_file")
    write_msg = execute_write_file("smoke_test.txt", "hello from smoke_test_tools")
    print(write_msg)

    workspace = Path(__file__).resolve().parent / "llm_workspace"
    out_file = workspace / "smoke_test.txt"
    print(f"File exists: {out_file.exists()} -> {out_file}")

    print_header("2) execute_run_command")
    cmd_msg = execute_run_command("python -c \"print('run_command ok')\"")
    print(cmd_msg)

    print_header("3) LangChain tools")
    tools = get_langchain_tools()
    write_tool = next(t for t in tools if t.name == "write_file")
    run_tool = next(t for t in tools if t.name == "run_command_in_docker")

    tool_write_result = write_tool.invoke(
        {"filename": "tool_smoke_test.txt", "content": "hello from LangChain tool"}
    )
    print("write_file tool ->", tool_write_result)

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
            "print(json.dumps({'reset_ok': r.get('ok'), 'step_ok': s.get('ok'), 'traj_path': t.get('path')}, ensure_ascii=False))\n"
            "PY"
        )
        env_api_result = execute_run_command(env_test_script)
        print(env_api_result)

    print_header("DONE")
    print("Smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
