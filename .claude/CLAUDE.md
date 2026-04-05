## Introduction to the Autumn Environment

The Autumn environment is a benchmark for testing the capabilities of LLMs in world-model learning. It contains several environments, and each environment has three tasks: masked frame prediction, change detection, and planning. The environments are grid worlds and usually contain hidden states, so they are modeled as POMDPs. I am currently focused on the planning task, where the agent needs to reach a goal state.

`MARAProtocol/python_examples/autumnbench/concrete_envs.py` contains examples of how to load the environments from Python.

`MARAProtocol/python_examples/autumnbench/example_benchmark/programs` contains the ground-truth world models. These world models are written in a new DSL called Autumn. Files named like `7XF97_change_detection_wrong_program.sexp` are used for change-detection tasks.

`MARAProtocol/python_examples/autumnbench/example_benchmark/prompts` contains task prompts. Files named `*_planning.json` define the target states for the corresponding environments. In these JSON files, the `goal` variable defines the target color for each grid cell. The color mapping is defined in `MARAProtocol/python_examples/autumnbench/example_benchmark/color_dict.yaml`, and the agent only needs to match the goal state in grid cells where the `mask` value is `1`.

`MARAProtocol/python_examples/autumnbench/example_benchmark/python_programs` contains Python implementations of the corresponding environments.

`MARAProtocol/python_examples/autumnbench/example_benchmark/python_programs_obfuscated` stores obfuscated versions of the Python implementations in which the object names are hidden. The obfuscation dictionary is stored in `MARAProtocol/python_examples/autumnbench/example_benchmark/obfuscation_mapping.json`.

I also generated planning-task variants for some environments. They are stored in `MARAProtocol/python_examples/autumnbench/example_benchmark/program_with_testcases`.


## Some Code Snippets for Interacting with Environments

`MARAProtocol/python_examples/autumnbench/langchain_utils.py` can be used to call LLMs through LangChain. It includes caching and usage logging.

Several files in the `explore_by_code` folder are important.

`env_api_server.py` provides an HTTP server for the Autumn environment. After starting this server, we can interact with the environment through a local port. It also provides an API for saving the current trajectory to a folder.

`env_wrapper.py` provides a Docker-based shell tool and wraps it as a LangChain tool. The Docker container can access the port opened by `env_api_server.py`, so programs running inside the container can also access the Autumn environment. The Docker runtime mounts a local folder created under `llm_workspace_runs`, and that folder is initialized from `llm_workspace`.

`llm_workspace/env_api_client.py` contains helper functions for querying local ports from inside Docker. Using these functions, programs running inside the container can access the Autumn environment service running outside the container.

`llm_workspace/check_traj_example.py` helps check a Python world model against saved trajectories.

`shell_react_explorer.py` contains a ReAct agent equipped with shell tools for environment exploration and code synthesis.

`baseline_optimizer.py` contains the core async logic for the baseline (fully autonomous) workflow optimization loop. It runs a designer LLM agent that writes a LangGraph workflow based on the instructions from the user and previous execution traces, then executes that workflow on K variants in parallel.