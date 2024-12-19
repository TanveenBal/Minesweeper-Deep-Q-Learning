import random
import numpy as np
import pandas as pd
from collections import deque

class MinesweeperEnv(object):
    def __init__(self, width, height, n_mines,
        rewards={'win':1, 'lose':-1, 'progress':0.3, 'guess':-0.3, 'no_progress' : -0.3}):
        self.nrows, self.ncols = width, height
        self.ntiles = self.nrows * self.ncols
        self.n_mines = n_mines
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()
        self.n_clicks = 0
        self.n_progress = 0
        self.n_wins = 0

        self.rewards = rewards

    def init_grid(self):
        board = np.zeros((self.nrows, self.ncols), dtype='object')
        mines = self.n_mines

        while mines > 0:
            row, col = random.randint(0, self.nrows-1), random.randint(0, self.ncols-1)
            if board[row][col] != 'B':
                board[row][col] = 'B'
                mines -= 1

        return board

    def get_neighbors(self, coord):
        x,y = coord[0], coord[1]

        neighbors = []
        for col in range(y-1, y+2):
            for row in range(x-1, x+2):
                if ((x != row or y != col) and
                    (0 <= col < self.ncols) and
                    (0 <= row < self.nrows)):
                    neighbors.append(self.grid[row,col])

        return np.array(neighbors)

    def count_bombs(self, coord):
        neighbors = self.get_neighbors(coord)
        return np.sum(neighbors=='B')

    def get_board(self):
        board = self.grid.copy()

        coords = []
        for x in range(self.nrows):
            for y in range(self.ncols):
                if self.grid[x,y] != 'B':
                    coords.append((x,y))

        for coord in coords:
            board[coord] = self.count_bombs(coord)

        return board

    def get_state_im(self, state):
        '''
        Gets the numeric image representation state of the board.
        This is what will be the input for the DQN.
        '''
        state_im = np.full((self.nrows, self.ncols), '-1.0', dtype='float32')

        for row in range(self.nrows):
            for col in range(self.ncols):
                value = state[row, col]
                if value == "U":
                    value = -1
                elif value == "B":
                    value = -2
                state_im[row][col] = np.float32(value / 8)

        return state_im

    def init_state(self):
        state = np.full((self.nrows, self.ncols), 'U', dtype='object')
        state_im = self.get_state_im(state)

        return state, state_im

    def color_state(self, value):
        if value == -1:
            color = 'white'
        elif value == 0:
            color = 'slategrey'
        elif value == 1:
            color = 'blue'
        elif value == 2:
            color = 'green'
        elif value == 3:
            color = 'red'
        elif value == 4:
            color = 'midnightblue'
        elif value == 5:
            color = 'brown'
        elif value == 6:
            color = 'aquamarine'
        elif value == 7:
            color = 'black'
        elif value == 8:
            color = 'silver'
        else:
            color = 'magenta'

        return f'color: {color}'

    def draw_state(self, state_im):
        state = state_im * 8.0
        state_df = pd.DataFrame(state.reshape((self.nrows, self.ncols)), dtype=np.int8)

        display(state_df.style.applymap(self.color_state))

    def click(self, row, col):
        # Convert action index to 2D coordinates
        coord = (row, col)

        value = self.board[coord]

        # Ensure the first move is not a bomb
        if value == 'B' and self.n_clicks == 0:
            # Pick a random safe tile
            while value == 'B':
                random_row = random.randint(0, self.nrows - 1)
                random_col = random.randint(0, self.ncols - 1)
                coord = (random_row, random_col)
                value = self.board[coord]

        # Update state with the clicked value
        self.state[coord] = value

        # Reveal neighbors if the clicked value is 0
        if value == 0:
            self.reveal_neighbors(coord)

        self.n_clicks += 1

    def reveal_neighbors(self, coord):
        visited = set()
        queue = deque()
        queue.append(coord)
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while queue:
            curr_row, curr_col = queue.popleft()

            # Skip if already visited
            if (curr_row, curr_col) in visited:
                continue

            visited.add((curr_row, curr_col))

            self.state[curr_row, curr_col] = self.board[curr_row, curr_col]

            if self.board[curr_row, curr_col] == 0:
                for dr, dc in directions:
                    adj_row, adj_col = curr_row + dr, curr_col + dc
                    
                    if 0 <= adj_row < self.nrows and 0 <= adj_col < self.ncols:
                        if (adj_row, adj_col) not in visited:
                            queue.append((adj_row, adj_col))


    def reset(self):
        self.n_clicks = 0
        self.n_progress = 0
        self.grid = self.init_grid()
        self.board = self.get_board()
        self.state, self.state_im = self.init_state()

    
    def step(self, row, col):
        done = False

        # Get the coordinates from the action index
        coord = (row, col)

        current_state = self.state_im.copy()

        # Click the tile
        self.click(row, col)

        new_state_im = self.get_state_im(self.state)
        self.state_im = new_state_im

        if self.state[coord] == 'B':  # Lose condition
            reward = self.rewards['lose']
            done = True

        elif np.sum(new_state_im == -0.125) == self.n_mines:  # Win condition
            reward = self.rewards['win']
            done = True
            self.n_progress += 1
            self.n_wins += 1

        elif np.sum(new_state_im == -0.125) == np.sum(current_state == -0.125):  # No progress
            reward = self.rewards['no_progress']

        else:  # Progress condition
            neighbors = self.get_neighbors(coord)
            if all(t==-0.125 for t in neighbors):
                reward = self.rewards['guess']
            else:
                reward = self.rewards['progress']
                self.n_progress += 1  # Track number of non-isolated clicks

        return self.state_im, reward, done
