import argparse
from tqdm import tqdm
from keras.models import load_model
from keras.losses import MeanSquaredError
from MinesweeperAgentWeb import *

def parse_args():
    parser = argparse.ArgumentParser(description="Play Minesweeper online using a DQN")
    parser.add_argument("--model", type=str, default="16x16_conv128x4_dense512x2_y0.1_minlr0.001",
                        help="name of model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to play")

    return parser.parse_args()

params = parse_args()

my_model = load_model(f"models/{params.model}.keras")

def main():
    pg.FAILSAFE = True
    agent = MinesweeperAgentWeb(my_model)

    for _ in tqdm(range(params.episodes)):
        agent.reset()
        agent.init_board()

        done = False
        while not done:
            # Try to evaluate the board first.
            performed_action = agent.evaluate_board()

            # If no moves from `evaluate_board`, fallback to model-based action.
            if not performed_action:
                action = agent.get_action()
                agent.step(action, uncover=True)
            done = agent.done

if __name__ == "__main__":
    main()
