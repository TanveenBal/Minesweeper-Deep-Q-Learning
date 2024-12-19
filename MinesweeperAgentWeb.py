import numpy as np
import pyautogui as pg
import pyscreeze as ps
from collections import deque

EPSILON = 0.01

CONFIDENCES = {
    "unsolved": 0.99,
    "flag": 0.90,
    "mine": 0.99,
    "0": 0.99,
    "1": 0.95,
    "2": 0.95,
    "3": 0.85,
    "4": 0.90,
    "5": 0.90,
    "6": 0.90,
    "7": 0.90,
    "8": 0.90
}


TILES = {
    "U": "unsolved",
    "F": "flag",
    "M": "mine",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
}

TILE_WIDTH = 16
TILE_LENGTH = 16

class MinesweeperAgentWeb(object):
    def __init__(self, model):
        self.solved = set()
        self.flagged = set()
        self.done = False
        pg.click((10,100)) # click on current tab so "F2" resets the game
        self.reset()

        self.mode, self.loc, self.dims = self.get_loc()
        self.nrows, self.ncols = self.dims[0], self.dims[1]
        self.ntiles = self.dims[2]
        self.board = self.state = None
        self.init_board()
        self.update_state()
        # self.new_update_state()
        self.epsilon = EPSILON
        self.model = model
        

    def reset(self):
        pg.press("f2")
        self.solved.clear()
        self.flagged.clear()
        self.done = False
        
    def get_loc(self):
        """
        obtain mode, screen coordinates and dimensions for Minesweeper board
        """

        modes = {"beginner":(8,8,64), "intermediate":(16,16,256), "expert":(16,30,480)}
        boards = {}
        for mode in modes.keys():
            try:
                boards[mode] = pg.locateOnScreen(f"pics/{mode}.png", confidence=.8)
            except pg.ImageNotFoundException:
                boards[mode] = None

        assert boards != {"beginner":None, "intermediate":None, "expert":None},\
            "Minesweeper board not detected on screen"

        for mode in boards.keys():
            if boards[mode] != None:
                diff = mode
                loc = boards[mode]
                dims = modes[mode]

        return diff, loc, dims
    
    def init_board(self):
        unsorted_tiles = list(pg.locateAllOnScreen(f"pics/unsolved.png", region=self.loc, confidence=CONFIDENCES["unsolved"]))

        tiles = []
        for coords in unsorted_tiles:
            tiles.append({"coord": (coords[0], coords[1]), "value": "U"})

        self.board = sorted(tiles, key=lambda x: (x["coord"][1], x["coord"][0]))
        

    def get_tile(self, coords):
        for tile_key, tile_name in TILES.items():
            try:
                result = list(pg.locateAllOnScreen(f"pics/{tile_name}.png", region=(coords[0]-1, coords[1]-1, TILE_WIDTH+1, TILE_LENGTH+1), confidence=CONFIDENCES[tile_name]))
                return tile_key
            except ps.ImageNotFoundException:
                continue
        self.done = True
        return "M"

    def get_tiles(self, tile, bbox):
        """
        Gets all locations of a given tile.
        Different confidence values are needed to correctly find different tiles with grayscale=True
        """
        conf = CONFIDENCES[tile]
        try:
            tiles = list(pg.locateAllOnScreen(f"pics/{tile}.png", region=bbox, confidence=conf))
        except ps.ImageNotFoundException:
            tiles = []

        return tiles

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

        # self.print_board()

    def print_board(self):
        board_2d = [tile["value"] for tile in self.board]
        board_2d = np.reshape(board_2d, (self.nrows, self.ncols))

        for row in board_2d:
            print(" ".join(str(cell) for cell in row))

    def update_state(self):
        """
        Updates the numeric image representation state of the board.
        This is what will be the input for the DQN.
        """

        state_im = [tile["value"] for tile in self.board]
        state_im = np.reshape(state_im, (self.nrows, self.ncols, 1)).astype(object)

        state_im[state_im=="U"] = -1
        state_im[state_im=="M"] = -2
        state_im[state_im=="F"] = -2

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        self.state = state_im

    def new_update_state(self):
        board_2d = [tile["value"] for tile in self.board]
        board_2d = np.reshape(board_2d, (self.nrows, self.ncols))
        numbers_layers = [np.zeros((self.nrows, self.ncols)) for _ in range(9)]
        uncovered_neighbors_layer = np.zeros((self.nrows, self.ncols))
        uncovered_layer = np.zeros((self.nrows, self.ncols))

        board_2d = np.reshape([tile["value"] for tile in self.board], (self.nrows, self.ncols))

        for row in range(self.nrows):
            for col in range(self.ncols):
                value = board_2d[row][col]

                if value == "U":
                    neighbors = [
                        (row + dr, col + dc)
                        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                        if 0 <= row + dr < self.nrows and 0 <= col + dc < self.ncols
                    ]
                    uncovered_neighbors_layer[row, col] = sum(1 for r, c in neighbors if board_2d[r][c] != "U" and board_2d[r][c] != "M")
                else:
                    if value == "M":
                        continue
                    numbers_layers[int(value)][row, col] = 1
                    uncovered_layer[row, col] = 1

        self.state = np.stack(numbers_layers + [uncovered_neighbors_layer, uncovered_layer], axis=-1).astype(np.float16)

    def new_get_action(self):
        board_input = self.state.reshape(1, self.nrows, self.ncols, self.state.shape[2])  # Including all layers
        unsolved = [i for i, tile in enumerate(self.board) if tile["value"] == "U"]  # Unsolved tiles

        rand = np.random.random()  # Exploration vs exploitation decision
        if rand < self.epsilon:
            print("Exploring: Random guess...")
            move = np.random.choice(unsolved)  # Randomly choose from unsolved tiles
        else:
            # Use the model to predict Q-values for all possible actions
            q_values = self.model.predict(board_input)  # Predict Q-values
            q_values = q_values.flatten()  # Flatten the output to match the action space

            # Mask solved or invalid tiles by setting their Q-values to negative infinity
            for i in range(len(q_values)):
                if i not in unsolved:
                    q_values[i] = -np.inf

            print(q_values)
            move = np.argmax(q_values)  # Choose the action with the highest Q-value

        return move

    def get_action(self):
        board_input = self.state.reshape(1, self.nrows, self.ncols, 1)
        unsolved = [i for i, tile in enumerate(self.board) if tile["value"] == "U"]

        rand = np.random.random()

        if rand < self.epsilon:
            print("Exploring: Random guess...")
            move = np.random.choice(unsolved)
        else:
            q_values = self.model.predict(board_input)  # Predict Q-values for all tiles
            q_values = q_values.flatten()  # Flatten the output to match action space

            for i in range(len(q_values)):
                if i not in unsolved:
                    q_values[i] = -np.inf

            move = np.argmax(q_values)  # Choose the action with the highest Q-value

        return move
    
    def step(self, action_index, uncover=False, flag=False):
        self.check_game_over()
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
            # self.new_update_state()

    def check_game_over(self):
        """
        Check if the game is won or lost.
        """
        try:
            if pg.locateOnScreen(f"pics/lost.png", region=self.loc, confidence=0.99):
                print("Game Over: Lost")
                self.done = True
        except pg.ImageNotFoundException:
            pass

        try:
            if pg.locateOnScreen(f"pics/won.png", region=self.loc, confidence=0.99):
                print("Game Over: Won")
                self.done = True
                pg.press('enter')
                pg.press('enter')
        except pg.ImageNotFoundException:
            pass


    def evaluate_board(self):
        """
        Evaluates the current board and takes appropriate actions for safe and mine tiles.
        Returns True if any actions (safe solve or mine flag) are performed, False otherwise.
        """
        performed_action = False

        for row in range(self.nrows):
            for col in range(self.ncols):
                if (row, col) in self.solved and (row, col) not in self.flagged:
                    safe = self.all_safe(row, col)
                    mines = self.all_mines(row, col)
                    if safe or mines:
                        performed_action = True

        # Return True if any action was taken
        return performed_action

    def all_safe(self, row, col):
        """
        Uncovers all safe neighbors around the cell at (row, col).
        Returns True if any tiles are safely uncovered, False otherwise.
        """
        action_taken = False

        # Safe if effective label is 0
        if self.effective_label(row, col) == 0:
            for neighbor_row, neighbor_col in self.get_neighbors(row, col):
                if (neighbor_row, neighbor_col) not in self.solved and (neighbor_row, neighbor_col) not in self.flagged:
                    # Uncover the tile
                    self.solve(neighbor_row, neighbor_col)
                    action_taken = True
        return action_taken


    def all_mines(self, row, col):
        """
        Flags all mine neighbors around the cell at (row, col).
        Returns True if any tiles are flagged as mines, False otherwise.
        """
        action_taken = False

        # Get neighbors and effective label
        neighbors = self.get_neighbors(row, col)
        unsolved_neighbors = [
            (n_row, n_col)
            for n_row, n_col in neighbors
            if (n_row, n_col) not in self.solved and (n_row, n_col) not in self.flagged
        ]

        # If all unsolved neighbors are mines
        if len(unsolved_neighbors) == self.effective_label(row, col):
            for n_row, n_col in unsolved_neighbors:
                self.flag(n_row, n_col)
                action_taken = True
        return action_taken
    
    def get_neighbors(self, row, col):
        neighbors = []
        dirs = [(1, 1), (0, 1), (-1, 1), (-1, 0), (-1 , -1), (0, -1), (1, -1), (1, 0)]
        for d_row, d_col in dirs:
            n_row = row + d_row
            n_col = col + d_col
            if 0 <= n_row < self.nrows and 0 <= n_col < self.ncols:
                neighbors.append((n_row, n_col))
        return neighbors

    def effective_label(self, row, col):
        """
        Calculates the effective label of the cell at (row, col),
        accounting for flagged neighbors as mines.
        """
        # Only calculate if the cell is solved
        if (row, col) in self.solved:
            index = row * self.ncols + col
            label = int(self.board[index]["value"])  # Ensure casting to integer
            neighbors = self.get_neighbors(row, col)

            # Subtract flagged neighbors
            for n_row, n_col in neighbors:
                if (n_row, n_col) in self.flagged:
                    label -= 1
            return label
        return 0

    def flag(self, row, col):
        self.step(row * self.ncols + col, flag=True)

    def solve(self, row, col):
        self.step(row * self.ncols + col, uncover=True)
