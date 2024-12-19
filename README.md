## I created an AI to play minesweeper for you!

This is different from just simply training a model on minesweeper through trial and error. As most people know, there are guaranteed moves for a player to make while playing the game. These rules are added on top of the model to make for guaranteed moves.

<table align="center">
  <tr>
    <td><img src="src/8x8.gif" alt="Pong"></td>
    <td><img src="src/16x16.gif" alt="Pong 16x16"></td>
    <td><img src="src/16x30.gif" alt="Pong 16x30"></td>
  </tr>
</table>

## Table of Contents
1. [Introduction to Minesweeper](#intro)
2. [Reinforcement Learning](#RL)
3. [Deep Q-Learning Networks](#DQN)
4. [Using AI to Beat Minesweeper](#AI)

### Introduction: The Game of Minesweeper <a name='intro'></a>

Minesweeper is a classic logic-based puzzle game that has been captivating players since its inception in 1989. The goal is simple: uncover all the tiles without triggering any mines. As you reveal tiles, numbers appear, indicating how many mines are adjacent to each uncovered tile. Using this information, players must deduce where it is safe to click next and avoid the mines hidden beneath the tiles.

**Is it possible for a computer to beat minesweeper?**

Of course, it's possible—but how exactly can it be done? While Minesweeper’s logical rules make it solvable through algorithms based on if-else conditions, the true challenge comes from teaching a computer to play the game with an evolving understanding. In this project, we do more than just rely on a machine learning model that learns from trial and error. Instead, the system is enhanced with built-in game rules—those guaranteed moves players can make based on number clues—allowing the AI to leverage known strategies alongside its learned decision-making. This combination of learned moves and guaranteed strategies pushes the boundaries of how an AI can approach Minesweeper.

This is where Reinforcement Learning steps in!

### What is Reinforcement Learning? <a name='RL'></a>

Reinforcement Learning (RL) is a branch of machine learning that teaches an agent (the computer) to make decisions through trial and error, with the ultimate goal of accomplishing a task. This process mirrors how a human learns a new game—without knowing any of the game’s rules initially, but learning what constitutes a good or bad move over time. For instance, in Minesweeper, the computer learns by identifying actions that lead to success (such as uncovering tiles without hitting mines) as well as those that result in failure (landing on a mine). Through this feedback loop, the agent gradually improves its decision-making. The core components of RL are as follows:

-  **The Reward Structure**: Instead of predefined rules, RL uses rewards and penalties to guide the agent’s learning. By assigning rewards for beneficial actions and penalties for detrimental ones, the agent learns to understand what leads to success and failure.
- **The Agent**: The agent is the learning entity—in this case, the computer. It takes actions within the environment based on its current understanding of which actions will maximize rewards or minimize penalties.
- **The Environment**: The environment represents the game itself. It is where the agent operates, and its state changes every time the agent takes an action. Each action results in a reward, based on the agent’s performance, and this reward is used to inform future decisions. The interaction between the environment’s state, the agent’s action, the reward, and the new state forms a transition, which is crucial for learning.

The goal of RL is for the Agent to learn an optimal policy, which means determining the best course of action to achieve the highest rewards over time. There are many different RL algorithms, and in this project, a Deep Q-learning Network (DQN) was employed to guide the agent toward mastering the task.
So the goal of RL is for the **Agent** to learn an optimal **policy** by pursuing actions that return the greatest reward. There are several different types of RL algorithms. In this project, I used a **Deep Q-learning Network** (DQN).

### What is Deep Q-learning? <a name='DQN'></a>

Deep Q-Learning is a method in Reinforcement Learning where an agent learns how to make decisions through trial and error. In traditional Q-Learning, an agent uses a table to track the values of state-action pairs, **Q-values**, but this becomes impractical with large or continuous state spaces. Deep Q-Learning uses a neural network to approximate the Q-value function.

The key steps in Deep Q-learning involve:

1. Choosing Actions: Actions are chosen based on the estimated Q-value for each possible action at a given state. This is typically done using the epsilon-greedy strategy (exploration vs. exploitation).
2. Learning: After taking an action, the agent receives a reward. The neural network learns by updating its Q-values based on these rewards and a discount factor that reduces the importance of future rewards.
3. Training: A target network is used to calculate the Q-values for next states, and periodically updated to improve stability.

Deep Q-Learning in Minesweeper

In this Minesweeper implementation, the state is represented by the board with values from `{-.25, 1}`, with `1` for unknown tiles, `-2` for mines, and for known tiles it's the tile's `label/8` (this is done for effecient computations when training). The agent’s task is to decide which tiles to reveal while avoiding mines. A **CNN** (convolutional neural network) is used to predict the best action at each state.

Here's a simple breakdown of what the agent does:

1. State Representation: The Minesweeper board is represented as an image, which allows the **CNN** to learn patterns about tile groups and their relationships (similar to how image recognition works).
2. Action Selection: The agent can either pick a random action (exploration) or the action with the highest Q-value (exploitation).
3. Experience Replay: The agent stores its experiences (state, action, reward, next state) in a memory buffer and samples from this buffer to break the correlation between consecutive training data and improve learning.
4. Target Network: The target network is periodically updated to prevent instability during training. It estimates the Q-values, which are used by the main network to update its own Q-values.

```python
def create_dqn(learn_rate, input_dims, n_actions, conv_units, dense_units):
    # Define the CNN architecture for the Q-learning model
    model = Sequential([
        Conv2D(conv_units, (3,3), activation='relu', padding='same', input_shape=input_dims),
        Conv2D(conv_units, (3,3), activation='relu', padding='same'),
        Conv2D(conv_units, (3,3), activation='relu', padding='same'),
        Conv2D(conv_units, (3,3), activation='relu', padding='same'),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dense(dense_units, activation='relu'),
        Dense(n_actions, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learn_rate, epsilon=1e-4), loss='mse')
    return model
```
- This is the function to create a deep Q-learning network (DQN) using convolutional layers followed by dense layers. This helps in learning patterns in the Minesweeper game from the board's state image.

```python
class DQNAgent(object):
    def __init__(self, env, model_name=MODEL_NAME, conv_units=CONV_UNITS, dense_units=DENSE_UNITS):
        self.env = env
        self.model = create_dqn(self.learn_rate, (self.env.state_im.shape[0], self.env.state_im.shape[1], 1), self.env.ntiles, conv_units, dense_units)
        self.target_model = create_dqn(self.learn_rate, (self.env.state_im.shape[0], self.env.state_im.shape[1], 1), self.env.ntiles, conv_units, dense_units)
        self.target_model.set_weights(self.model.get_weights())  # Initialize target model weights to be the same as the model weights.
        self.replay_memory = deque(maxlen=MEM_SIZE)  # Memory buffer for experiences
        self.target_update_counter = 0
```
- Here, the `DQNAgent` class is initialized with the environment and model parameters. The `agent` sets up both the `model` and `target model`. It uses a `deque` to store experiences, which are later sampled for training.

```python
def get_action(self, state):
    # Randomly choose a move with probability epsilon (explore) or choose the best action (exploit)
    rand = np.random.random()  # Random value between 0 and 1
    if rand < self.epsilon:
        move = np.random.choice(unsolved)  # Explore: choose a random move
    else:
        moves = self.model.predict(np.reshape(state, (1, self.env.nrows, self.env.ncols, 1)))  # Exploit: choose the best action from the current Q-values
        move = np.argmax(moves)
    return move
```
- This method selects an action based on the epsilon-greedy strategy. If the agent is in the exploration phase, it picks a random move. Otherwise, it picks the best move according to the learned Q-values.

```python
def train(self, done):
    if len(self.replay_memory) < MEM_SIZE_MIN:  # Not enough memory to train
        return
    
    batch = random.sample(self.replay_memory, BATCH_SIZE)  # Sample a batch of experiences
    
    current_states = np.array([transition[0] for transition in batch])  # Get current states from the batch
    current_qs_list = self.model.predict(current_states)
    
    new_current_states = np.array([transition[3] for transition in batch])  # Get next states
    future_qs_list = self.target_model.predict(new_current_states)
    
    X, y = [], []
    for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
        # Calculate new Q-value for the current state
        if not done:
            max_future_q = np.max(future_qs_list[i])  # Max Q-value of next state
            new_q = reward + DISCOUNT * max_future_q  # Update Q-value using Bellman equation
        else:
            new_q = reward  # For terminal states, no future rewards
        
        # Update the Q-values list
        current_qs = current_qs_list[i]
        current_qs[action] = new_q

        # Append the updated states to the training data
        X.append(current_state)
        y.append(current_qs)

    # Fit the model to the training data
    self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, shuffle=False)

    if done:
        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())  # Update target model's weights
            self.target_update_counter = 0
```
- This `train` function performs the training of the Deep Q-learning model. It uses a batch of experiences from the replay memory and updates the model using the Bellman equation, also periodically updating the target model.

This approach helps the agent avoid overfitting and improves its ability to generalize to new game states. By combining experience replay, target networks, and convolutional layers, the agent learns a better policy for solving Minesweeper puzzles.

## Using the AI to Beat Minesweeper <a name='AI'></a>

My MinesweeperAgent uses a Convolutional Neural Network (CNN) and Q-learning and garunteed Minesweeper rules to autonomously play and solve Minesweeper. The agent operates in a state space where the board is represented numerically, and uses a model to predict the optimal move based on past experiences. The key aspect of the system is integrating computer vision for board state detection and AI for decision-making.

```python
# (rows, columns, mines) dimensions and mine number for Beginner mode
agent = MinesweeperAgent(8, 8, 10)
```

### Board State Representation

The board's state is represented as an "image" where each tile is:

- 0: Empty tile,
- 1-8: Numbered tiles indicating the number of adjacent mines,
- U: Unknown (unsolved) tile,
- F: Tile marked as a mine with flag.
- M: Tile that is a mine.

These tile states are scaled to values between -1 and 1 for better input processing into the CNN:

```python
state_im[state_im == "U"] = -1
state_im[state_im == "M"] = -2
state_im[state_im == "F"] = -2
state_im = state_im.astype(np.int8) / 8
```

- The state image is reshaped and normalized before being passed to the model.

### Model Architecture

The CNN model is composed of four convolutional layers (each with 128 neurons) followed by two fully connected layers (each with 512 neurons). This architecture was chosen after experimentation and comparison with other models based on training efficiency:

```python
# Simple DQN architecture with convolutional layers
moves = self.model.predict(state.reshape((1, self.nrows, self.ncols, 1)))
```

Here’s how the model handles actions during gameplay:

```python
def get_action(self, state):
    board = state.reshape(1, self.ntiles)
    unsolved = [i for i, x in enumerate(board[0]) if x == -0.125]

    rand = np.random.random()
    if rand < self.epsilon:  # Random move (explore)
        move = np.random.choice(unsolved)
    else:
        moves = self.model.predict(state.reshape((1, self.nrows, self.ncols, 1)))
        moves[board != -0.125] = np.min(moves)  # Mask already revealed tiles
        move = np.argmax(moves)

    return move
```
- The epsilon-greedy strategy balances exploration (random moves) and exploitation (using the model’s learned predictions).
- The model focuses on solving only unsolved tiles by applying masking on previously revealed ones to avoid unnecessary actions.

### Computer Vision Integration with PyAutoGUI

To control the Minesweeper board, the agent uses the `PyAutoGUI` library for mouse control and screen reading. The board is detected using predefined screenshots of tile types (e.g., unsolved, flagged, mine) to extract coordinates of each tile:

The `MinesweeperAgentWeb` object must initialize the board using `init_board` which gathers the coordinates of each tile using `PyAutoGUI`.
```python
def init_board(self):
    unsorted_tiles = list(pg.locateAllOnScreen(f"pics/unsolved.png", region=self.loc, confidence=CONFIDENCES["unsolved"]))

    tiles = []
    for coords in unsorted_tiles:
        tiles.append({"coord": (coords[0], coords[1]), "value": "U"})

    self.board = sorted(tiles, key=lambda x: (x["coord"][1], x["coord"][0]))
```

After making an action, e.g. flagging or uncovering a tile the `get_tile` function is called to fetch the type of tile it is:
```python
def get_tile(self, coords):
    for tile_key, tile_name in TILES.items():
        try:
            result = list(pg.locateAllOnScreen(f"pics/{tile_name}.png", region=(coords[0]-1, coords[1]-1, TILE_WIDTH+1, TILE_LENGTH+1), confidence=CONFIDENCES[tile_name]))
            return tile_key
        except ps.ImageNotFoundException:
            continue
    self.done = True
    return "M"
```

A complication occurs when updating the board. If the tile is `empty` we need to cascade and know the values of the tiles adjacent to the `empty` tile. This is done effeciently using Breath First Search `BFS`:
```python
    def update_board(self, action_index, action_coords):
        """
        Gets the state of the board as a dictionary of coordinates and values,
        ordered from left to right, top to bottom
        """
        self.check_game_over()
        if self.done:
            return
        
        tile_value = self.get_tile(action_coords)
        self.board[action_index]["value"] = tile_value

        
        if tile_value == "0": # Can make this bfs instead of what it does right now, basically bfs until there are unsolveds for effeciency
            self.board = np.reshape(self.board, (self.nrows, self.ncols))
            visited = set()
            queue = deque()
            action_row, action_col = divmod(action_index, self.ncols)
            queue.append((action_row, action_col))
            self.solved.remove((action_row, action_col))

            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            
            while queue:
                curr_row, curr_col = queue.popleft()

                # Skip if already visited
                if (curr_row, curr_col) in visited or (curr_row, curr_col) in self.solved:
                    continue

                visited.add((curr_row, curr_col))
                self.solved.add((curr_row, curr_col))
                curr_coords = self.board[curr_row][curr_col]["coord"]

                # Get the value of the current tile
                tile_value = self.get_tile(curr_coords)
                self.board[curr_row][curr_col]["value"] = tile_value

                if tile_value.isdigit():
                    # If tile is also '0', add neighbors to the queue
                    if tile_value == "0":
                        for dr, dc in directions:
                            new_row, new_col = curr_row + dr, curr_col + dc
                            if 0 <= new_row < self.nrows and 0 <= new_col < self.ncols:
                                if (new_row, new_col) not in visited:
                                    queue.append((new_row, new_col))
            self.board = self.board.ravel()
```



This method provides accurate tile identification, enabling the agent to take precise actions.

### Step Execution and State Updates

For each move, the agent performs one of the following actions based on its decision-making:

1. Uncovering a tile: If the move results in uncovering a tile, pg.click() simulates a mouse click.
2. Flagging a mine: If a tile is suspected to contain a mine, pg.rightClick() is used to mark it.
3. Updating board state: After each move, the board state is updated to reflect the newly uncovered or flagged tile.

Here’s how the `step` function works:
```python
def step(self, action_index, uncover=False, flag=False):
    action_row, action_col = divmod(action_index, self.ncols)
    action_coords = self.board[action_index]["coord"]
    if uncover:
        pg.click(action_coords)
        self.solved.add((action_row, action_col))
    elif flag:
        pg.rightClick(action_coords)
        self.flagged.add((action_row, action_col))
    else:
        raise AssertionError

    if not self.done:
        self.update_board(action_index, action_coords)
        self.update_state()
```

After each action, the agent re-checks the state of the board (update_board() and update_state()) to see the effect of the move.
Board Updates and Cascading Effect

Whenever a tile with a value of 0 (indicating no adjacent mines) is revealed, the agent cascades through adjacent tiles, automatically uncovering more tiles until it hits non-zero values or already revealed tiles.

### Game Over Detection

Lastly, the agent continuously checks for a "game over" state by identifying a loss screen using pg.locateOnScreen():
```python
def check_game_over(self):
    if pg.locateOnScreen(f"pics/lost.png", region=self.loc, confidence=0.99):
        print("Game Over...")
        self.done = True
```
- This is a key part of the loop that ensures the agent knows when to stop acting.

## Work in progress
- Hyperparamater tuning
- Fixing the tensorboard data readings (something wrong with DQN possibly)
- Adding the rules to the environment so that it only learns based on boards with more advanced patterns or probabilities needed (improving the overall model hopefully)
- Improve time it takes for `PyAutoGUI` to detect all tiles


Found something wrong in my code or have questions? Feel free to contact me:
- tanveenbal@gmail.com
- [my linkedin](https://www.linkedin.com/in/tanveenbal/)
