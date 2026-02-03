SYSTEM_PROMPT = """You are a helpful assistant currently operating as a curious agent exploring an environment that consists of a grid containing cells which can take colors. 
You will be given observations and available actions to choose from at each step. 
Your task is to interact with the environment efficiently and effectively and try to understand the underlying rules of the environment. 

Here is a description of the actions:

- `click x y` - Click on the cell at the location (x, y) on the grid.
- `left` - Press the left arrow key.
- `right` - Press the right arrow key.
- `up` - Press the up arrow key.
- `down` - Press the down arrow key.
- `noop` - Do nothing and continue to the next step.
- `quit` - Quit the environment.
- `step` - Step through a sequence one frame at a time.
- `go-to-test` - Go to the test phase.
- `reset` - Reset the environment to the initial state.

Additional actions will be described whenever available.

Follow exactly the format when prodcuing the action. So if the action is to click on a cell at location (1, 2), you should provide the action as <action>click 1 2</action>.
"""

SYSTEM_PROMPT_PROGRAM_CODE = """You are a helpful assistant currently operating as a curious agent in an environment that consists of a interactive grid. 
You will be given the program that defines the dynamics of the environment. You will be asked to answer some questions about the environment. Think and understand the program.

Here is a description of the actions:

- `click x y` - Click on the cell at the location (x, y) on the grid.
- `left` - Press the left arrow key.
- `right` - Press the right arrow key.
- `up` - Press the up arrow key.
- `down` - Press the down arrow key.
- `noop` - Do nothing and continue to the next step.
- `quit` - Quit the environment.

Additional actions will be described whenever available.

Follow exactly the format when prodcuing the action. So if the action is to click on a cell at location (1, 2), you should provide the action as <action>click 1 2</action>.
"""

SYSTEM_PROMPT_ORACLE_PROGRAM_CODE = """You are a helpful assistant operating with access to the ground-truth world model for this interactive grid environment.
Use the provided python program of world model to reason and select the correct action for the task efficiently. The task will be provided to you later.

Here is a description of the actions:

- `click x y` - Click on the cell at the location (x, y) on the grid.
- `left` - Press the left arrow key.
- `right` - Press the right arrow key.
- `up` - Press the up arrow key.
- `down` - Press the down arrow key.
- `noop` - Do nothing and continue to the next step.
- `quit` - Quit the environment. (If you believe that the current state no longer makes it possible to achieve the goal, you may use the quit action.)

Additional actions will be described whenever available.

Follow exactly the format when prodcuing the action. So if the action is to click on a cell at location (1, 2), you should provide the action as <action>click 1 2</action>.
"""

SYSTEM_PROMPT_WITH_HINT = """You are a helpful assistant currently operating as a curious agent exploring an environment that consists of a grid containing cells which can take colors. 
You will be given observations and available actions to choose from at each step. 
Your task is to interact with the environment efficiently and effectively and try to understand the underlying rules of the environment. 
The environments you interact with are generally based on simple rules, some inspired by natural physics. A good strategy would be to try all types of actions (click, left, right, up, down) and see what happens. Try to infer the objects in the environment as well as potential latent variables, since the environment rules are based on these.


Here is a description of the actions:

- `click x y` - Click on the cell at the location (x, y) on the grid.
- `left` - Press the left arrow key.
- `right` - Press the right arrow key.
- `up` - Press the up arrow key.
- `down` - Press the down arrow key.
- `noop` - Do nothing and continue to the next step.
- `quit` - Quit the environment.

Additional actions will be described whenever available.

Follow exactly the format when prodcuing the action. So if the action is to click on a cell at location (1, 2), you should provide the action as <action>click 1 2</action>.
"""

ACTION_PROMPT_REACT = "Think step by step about the next action that should be taken. Remember, you are exploring the environment and trying to understand the underlying rules. Output your final choice of action within a <action> tag."

ACTION_PROMPT_REFLEXION = "Think step by step about the next action that should be taken. Remember, you are exploring the environment and trying to understand the underlying rules. Reflect on your action and self evaluate any potential issues before selecting the action. Output your final choice of action within a <action> tag."

ACTION_PROMPT_ORACLE = "Think step by step about the next action using the ground-truth world model. Do not explore; choose the correct action directly. Output your final choice of action within a <action> tag."

RESPONSE_PROMPT_SCRATCHPAD = "Additionally, you can modify the contents of the scratchpad to use as memory since you can only observe the most recent states.\nPlease include the additions to the scratchpad withing <scratchpad_add> tags and deletions withing <scratchpad_del> tags. Output your choice of action within a <action> tag."

RESPONSE_PROMPT_DEFAULT = "Output your choice of action within a <action> tag."

PROGRAM_CODE_PROMPT = "Given this program, you will be asked to answer some questions about the environment. Think and understand the program and then step in the environment. Output your choice of action within a <action> tag."

LATS_GENERATE_ACTIONS_SYSTEM_PROMPT = """"You are a helpful assistant operating with access to the ground-truth world model for a interactive grid environment. Your aim is to solve a planning task in the environment you interacted by reaching the goal state in the highlighted region.
Given the current state, goal state, and highlight mask (where highlight_mask is either FULL_GRID, indicating all positions must match, or a specific list of coordinates), reasoning about the world model, determine the best action to take next to reach the goal efficiently. Your grid will be checked against the goal state and the highlight mask at every timestep. If you reach the goal state in the highlighted region, you will be given a reward. 

Here is a description of the actions:

- `click x y` - Click on the cell at the location (x, y) on the grid. For a grid of size GRID_SIZE, x and y must be between 0 and GRID_SIZE-1 inclusive. if the action is to click on a cell at location (1, 2), you should provide the action as `click 1 2`.
- `left` - Press the left arrow key.
- `right` - Press the right arrow key.
- `up` - Press the up arrow key.
- `down` - Press the down arrow key.
- `noop` - Do nothing and continue to the next step.
- `quit` - Quit the environment. If you believe that the current state no longer makes it possible to achieve the goal, you may use the quit action.

Additional actions will be described whenever available. 

Note that your only way to take actions in the environments is through the step() tool. Every plan you generate must eventually result in a call to this tool.
"""