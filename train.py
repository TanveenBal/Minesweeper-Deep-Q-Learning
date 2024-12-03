import argparse, pickle
from tqdm import tqdm
from keras.models import load_model
from DQN_agent import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# intake MinesweeperEnv parameters, beginner mode by default
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=8,
                        help='width of the board')
    parser.add_argument('--height', type=int, default=8,
                        help='height of the board')
    parser.add_argument('--n_mines', type=int, default=10,
                        help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=100_000,
                        help='Number of episodes to train on')

    return parser.parse_args()

params = parse_args()

AGG_STATS_EVERY = 100 # calculate stats every 100 games for tensorboard
SAVE_MODEL_EVERY = 1000 # save model and replay every 10,000 episodes

def main():
    env = MinesweeperEnv(params.width, params.height, params.n_mines)
    agent = DQNAgent(env, MODEL_NAME)

    progress_list, wins_list, ep_rewards = [], [], []
    n_clicks = 0

    # Attempt to load previous model and replay memory
    try:
        model_path = f'models/{MODEL_NAME}.keras'
        replay_path = f'replay/{MODEL_NAME}.pkl'

        if os.path.exists(model_path):
            agent.model = load_model(model_path)
            agent.target_model.set_weights(agent.model.get_weights())
            print(f"Loaded model from {model_path}")

        if os.path.exists(replay_path):
            with open(replay_path, 'rb') as f:
                agent.replay_memory = pickle.load(f)
                print(f"Loaded replay memory from {replay_path}")
    except Exception as e:
        print(f"Error loading model or replay memory: {e}")

    for episode in tqdm(range(1, params.episodes+1), unit='episode'):
        agent.tensorboard.step = episode

        env.reset()
        episode_reward = 0
        past_n_wins = env.n_wins

        done = False
        while not done:
            current_state = env.state_im

            action = agent.get_action(current_state)

            new_state, reward, done = env.step(action)

            episode_reward += reward

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)

            n_clicks += 1

        progress_list.append(env.n_progress)
        ep_rewards.append(episode_reward)

        if env.n_wins > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if len(agent.replay_memory) < MEM_SIZE_MIN:
            continue

        if not episode % AGG_STATS_EVERY:
            med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
            win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
            med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)

            agent.tensorboard.update_stats(
                progress_med = med_progress,
                winrate = win_rate,
                reward_med = med_reward,
                learn_rate = agent.learn_rate,
                epsilon = agent.epsilon)

            print(f'Episode: {episode}, Median progress: {med_progress}, Median reward: {med_reward}, Win rate : {win_rate}')

            if not episode % SAVE_MODEL_EVERY:
                os.makedirs('replay', exist_ok=True)  # Ensure directory exists
                try:
                    with open(f'replay/{MODEL_NAME}.pkl', 'wb') as output:
                        pickle.dump(agent.replay_memory, output)
                    
                    # Save model using the native Keras format
                    agent.model.save(f'models/{MODEL_NAME}.keras')
                except Exception as e:
                    print(f"Error saving model or replay memory: {e}")

if __name__ == "__main__":
    main()
