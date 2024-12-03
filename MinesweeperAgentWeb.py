import numpy as np
import pyautogui as pg
import pyscreeze as ps
from collections import deque

EPSILON = 0.01

CONFIDENCES = {
    "unsolved": 0.99,
    "mine": 0.99,
    "0": 0.99,
    "1": 0.90,
    "2": 0.90,
    "3": 0.80,
    "4": 0.90,
    "5": 0.90,
    "6": 0.90,
    "7": 0.90,
    "8": 0.90
}


TILES = {
    "U": "unsolved",
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
        pg.click((10,100)) # click on current tab so "F2" resets the game
        self.reset(initial=True)

        self.mode, self.loc, self.dims = self.get_loc()
        self.nrows, self.ncols = self.dims[0], self.dims[1]
        self.ntiles = self.dims[2]
        self.board = self.state = None
        self.init_board()
        self.update_state()
        self.solved = set()

        self.epsilon = EPSILON
        self.model = model
        self.done = False

    def reset(self, initial=False):
        pg.press("f2")

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

        # i = 0
        # for row in range(self.nrows):
        #     for col in range(self.ncols):
        #         self.board[i]["index"] = (row, col)
        #     i += 1
        

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
            self.solved.remove(action_index)

            directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            
            while queue:
                curr_row, curr_col = queue.popleft()

                # Skip if already visited
                if (curr_row, curr_col) in visited:
                    continue

                visited.add((curr_row, curr_col))
                curr_coords = self.board[curr_row][curr_col]["coord"]
                curr_index = (curr_row * self.ncols) + curr_col

                # Get the value of the current tile
                tile_value = self.get_tile(curr_coords)
                self.board[curr_row][curr_col]["value"] = tile_value

                if tile_value.isdigit():
                    self.solved.add(curr_index)  # Mark as solved

                    # If tile is also '0', add neighbors to the queue
                    if tile_value == "0":
                        for dr, dc in directions:
                            new_row, new_col = curr_row + dr, curr_col + dc
                            if 0 <= new_row < self.nrows and 0 <= new_col < self.ncols:
                                if (new_row, new_col) not in visited:
                                    queue.append((new_row, new_col))

            # for i in range(len(self.board)):
            #     if i not in self.solved:
            #         tile_value = self.get_tile(self.board[i]["coord"])
            #         self.board[i]["value"] = tile_value
            #         if tile_value.isdigit():
            #             self.solved.add(i)

            self.board = self.board.ravel()
        self.print_board()

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

        state_im = state_im.astype(np.int8) / 8
        state_im = state_im.astype(np.float16)

        self.state = state_im

    def get_action(self):
        board = self.state.reshape(1, self.ntiles)
        unsolved = [i for i, x in enumerate(board[0]) if x==-0.125]

        rand = np.random.random() # random value b/w 0 & 1

        if rand < self.epsilon: # random move (explore)
            print("guessing...")
            move = np.random.choice(unsolved)
        else:
            moves = self.model.predict(np.reshape(self.state, (1, self.nrows, self.ncols, 1)))
            moves[board != -0.125] = -np.inf
            move = np.argmax(moves)

        return move
    
    def step(self, action_index):
        # number of solved tiles prior to move (initialized at 0)
        #self.n_solved = self.n_solved_

        # get neighbors before clicking
        # neighbors = self.get_neighbors(action_index)

        self.check_game_over()
        action_coords = self.board[action_index]["coord"]
        pg.click(action_coords)
        pg.click((10, 100))
        self.solved.add(action_index)
        

        if not self.done:
            self.update_board(action_index, action_coords)
            self.update_state()

        return self.state, self.done

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
        except pg.ImageNotFoundException:
            pass

        self.done = False


