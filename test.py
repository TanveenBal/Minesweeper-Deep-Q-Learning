import argparse
from tqdm import tqdm
from keras.models import load_model
from keras.losses import MeanSquaredError
from MinesweeperAgentWeb import *

custom_objects = {
    "mse": MeanSquaredError()
}

def parse_args():
    parser = argparse.ArgumentParser(description='Play Minesweeper online using a DQN')
    parser.add_argument('--model', type=str, default='conv64x4_dense512x2_y0.1_minlr0.001_8x8',
                        help='name of model')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to play')

    return parser.parse_args()

params = parse_args()

my_model = load_model(f'models/{params.model}.h5', custom_objects=custom_objects)

def main():
    pg.FAILSAFE = True
    agent = MinesweeperAgentWeb(my_model)

    for _ in tqdm(range(1, params.episodes+1)):
        agent.reset()

        done = False
        while not done:
            try:
                action = agent.get_action()
                state, done = agent.step(action)
            except ValueError:
                break

if __name__ == "__main__":
    main()
