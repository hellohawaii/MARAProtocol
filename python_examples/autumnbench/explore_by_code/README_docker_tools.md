# Docker + FastAPI Environment Wrapping (explore_by_code)

## Added Components

- `env_api_server.py`: exposes environment APIs on the host machine
  - `POST /reset`
  - `POST /step`
  - `POST /save_trajectory`
- `llm_workspace/env_api_client.py`: used by Python code inside Docker to call the APIs above
- `env_wrapper.py`:
  - `execute_write_file(filename, content)`
  - `execute_run_command(command, timeout_seconds=15, ...)`
  - `get_langchain_tools(...)` returns two LangChain `StructuredTool`s

## 1) Start the host environment API

Run from the repository root:

```bash
python python_examples/autumnbench/explore_by_code/env_api_server.py
```

The API server dependencies are expected to come from the outer environment
(for example, already installed via `AutumnWeb/requirements.txt` in the outer Docker image).

By default, it listens on `0.0.0.0:8000`.

## 2) Access the environment API inside Docker

Use DinD (Docker daemon runs inside your current dev container). In this mode,
`execute_run_command` mounts the in-container path directly, so no host path remapping is needed.

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
print(env.reset(env_name='7XF97', seed=0).keys())
print(env.step('noop').keys())
print(env.save_trajectory('demo_traj').get('path'))
PY
```

Trajectories are saved to:

- `python_examples/autumnbench/explore_by_code/llm_workspace/traj/<ENV>/...json`

## 3) LangChain tools integration

```python
from python_examples.autumnbench.explore_by_code.env_wrapper import get_langchain_tools

tools = get_langchain_tools(
  docker_image='python:3.11-slim',
    timeout_seconds=15,
)
# tools[0] = write_file
# tools[1] = run_command_in_docker
```

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
