import os, sys

ROOT = os.getcwd()
sys.path.insert(1, f"{os.path.dirname(ROOT)}")

import warnings
warnings.filterwarnings("ignore")

from collections import deque
# from MinesweeperEnv import *
from minesweeper_env import *
from tensorboard import *
from DQN import *

# Environment settings
MEM_SIZE = 50_000 # number of moves to store in replay buffer
MEM_SIZE_MIN = 1_000 # min number of moves in replay buffer

# Learning settings
BATCH_SIZE = 64
LEARN_RATE = 0.01
LEARN_DECAY = 0.99975
LEARN_MIN = 0.001
DISCOUNT = 0.1 #gamma

# Exploration settings
epsilon = 0.95
EPSILON_DECAY = 0.99975
EPSILON_MIN = 0.01

# DQN settings
CONV_UNITS = 128 # number of neurons in each conv layer
DENSE_UNITS = 512 # number of neurons in fully connected dense layer
UPDATE_TARGET_EVERY = 5

# Default model name
MODEL_NAME = f"16x30_conv{CONV_UNITS}x4_dense{DENSE_UNITS}x2_y{DISCOUNT}_minlr{LEARN_MIN}"

class DQNAgent(object):
    def __init__(self, env, model_name=MODEL_NAME, conv_units=CONV_UNITS, dense_units=DENSE_UNITS):
        self.env = env

        # Deep Q-learning Parameters
        self.discount = DISCOUNT
        self.learn_rate = LEARN_RATE
        self.epsilon = epsilon
        self.model = create_dqn(
            self.learn_rate, (self.env.state_im.shape[0], self.env.state_im.shape[1], 1), self.env.ntiles, conv_units, dense_units)

        # target model - this is what we predict against every step
        self.target_model = create_dqn(
            self.learn_rate, (self.env.state_im.shape[0], self.env.state_im.shape[1], 1), self.env.ntiles, conv_units, dense_units)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs\\{model_name}", profile_batch=0)

    def get_action(self, state):
        unsolved = []
        for row in range(self.env.nrows):
            for col in range(self.env.ncols):
                if state[row, col] == -0.125: 
                    unsolved.append((row, col))
        
        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon:
            move = np.random.choice(range(len(unsolved)))
            row, col = unsolved[move]
            return row, col
        else:
            # Make predictions using the model
            moves = self.model.predict(np.reshape(state, (1, self.env.nrows, self.env.ncols, 1)))
            for row in range(self.env.nrows):
                for col in range(self.env.ncols):
                    if state[row, col] != -0.125: 
                        moves[0, row * self.env.ncols + col] = -np.inf
            move = np.argmax(moves)
            return np.unravel_index(move, (self.env.nrows, self.env.ncols))

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done):
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        current_states = np.array([transition[0] for transition in batch])
        current_states = np.reshape(current_states, (BATCH_SIZE, self.env.nrows, self.env.ncols, 1))
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        new_current_states = np.reshape(new_current_states, (BATCH_SIZE, self.env.nrows, self.env.ncols, 1))
        future_qs_list = self.target_model.predict(new_current_states)

        X,y = [], []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action[0] * self.env.ncols + action[1]] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE,
                       shuffle=False, verbose=0, callbacks=[self.tensorboard]\
                       if done else None)

        # updating to determine if we want to update target_model yet
        if done:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        # decay learn_rate
        self.learn_rate = max(LEARN_MIN, self.learn_rate*LEARN_DECAY)

        # decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)

if __name__ == "__main__":
    DQNAgent(MinesweeperEnv(8,8,10))
