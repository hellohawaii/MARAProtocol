# Docker + FastAPI Environment Wrapping (explore_by_code)

## Added Components

- `env_api_server.py`: exposes environment APIs on the host machine
  - `POST /reset`
  - `POST /step`
  - `POST /save_trajectory`
- `llm_workspace/env_api_client.py`: used by Python code inside Docker to call the APIs above
- `env_wrapper.py`:
  - `execute_run_command(command, timeout_seconds=15, ...)`
  - `get_langchain_tools(...)` returns one LangChain `StructuredTool` (`run_command_in_docker`)

## 1) Start the host environment API

Run from the repository root:

```bash
python python_examples/autumnbench/explore_by_code/env_api_server.py
```

The API server dependencies are expected to come from the outer environment
(for example, already installed via `AutumnWeb/requirements.txt` in the outer Docker image).

By default, it listens on `0.0.0.0:8001`.

## 2) Access the environment API inside Docker

Use DinD (Docker daemon runs inside your current dev container). In this mode,
`execute_run_command` uses Docker SDK and mounts the in-container path directly,
so no host path remapping is needed.

If Docker daemon is not running yet in the dev container, start it first:

```bash
dockerd --host=unix:///var/run/docker.sock --storage-driver=vfs
```

Then run the host API server (same container, another shell):

```bash
python python_examples/autumnbench/explore_by_code/env_api_server.py
```

Container-side example (this command can be executed via `execute_run_command`):

```bash
python - <<'PY'
from env_api_client import RemoteEnvWrapper

env = RemoteEnvWrapper()
print(type(env.reset()).__name__)             # dict state
print(type(env.step('noop')).__name__)             # dict state
print(env.save_trajectory('demo_traj'))            # success flag object
PY
```

Trajectories are saved to:

- `python_examples/autumnbench/explore_by_code/llm_workspace/traj/...json`

## 3) LangChain tools integration

```python
from python_examples.autumnbench.explore_by_code.env_wrapper import get_langchain_tools

tools = get_langchain_tools(
    docker_image='python:3.11-slim',
    timeout_seconds=15,
)
# tools[0] = run_command_in_docker
```

Optional: build a custom shell image from Dockerfile before first execution.

```python
tools = get_langchain_tools(
    timeout_seconds=20,
    dockerfile_path='python_examples/autumnbench/explore_by_code/Dockerfile.tool',
    docker_build_context='python_examples/autumnbench/explore_by_code',
)
```

`execute_run_command` keeps a persistent container per Python process.
This means shell state can be reused across multiple commands.

## 4) One-shot smoke test

Run from `/app/MARAProtocol`:

```bash
python python_examples/autumnbench/explore_by_code/smoke_test_tools.py
```

Optional: also test `env.reset()` / `env.step()` / `env.save_trajectory()` from inside Docker
(requires host API server started as in section 1):

```bash
python python_examples/autumnbench/explore_by_code/smoke_test_tools.py --test-env-api --env-name 7XF97
```

## 5) Simple ReAct agent with shell tool

You can run a standalone ReAct agent that only has one tool:
`run_command_in_docker`.

The agent is guided by a fixed system prompt and will decide by itself how to:
- explore the env via `RemoteEnvWrapper`,
- save trajectories,
- write Python model files,
- run `check_traj_example.py`,
- and stop when it decides understanding is sufficient.

Run:

```bash
python python_examples/autumnbench/explore_by_code/shell_react_explorer.py 7XF97 \
  --max-turns 120 \
  --timeout-seconds 30 \
  --save-transcript-path python_examples/autumnbench/explore_by_code/llm_workspace/agent_runs/7XF97_run.json
```

Optional flags:
- `--llm-model <model_id>`: defaults to `google/gemini-2.5-pro`
- `--dockerfile-path` and `--docker-build-context`: custom tool image build
- `--env-api-base-url`: defaults to `http://host.docker.internal:8001`
