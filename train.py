import argparse, pickle
from tqdm import tqdm
from keras.models import load_model
from DQN_agent import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DQN to play Minesweeper')
    parser.add_argument('--width', type=int, default=16, help='Width of the board')
    parser.add_argument('--height', type=int, default=30, help='Height of the board')
    parser.add_argument('--n_mines', type=int, default=99, help='Number of mines on the board')
    parser.add_argument('--episodes', type=int, default=100_000, help='Number of episodes to train on')
    return parser.parse_args()

params = parse_args()
AGG_STATS_EVERY = 500
SAVE_MODEL_EVERY = 500

def save_checkpoint(agent, episode, progress_list, wins_list, ep_rewards):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'episode': episode,
        'progress_list': progress_list,
        'wins_list': wins_list,
        'ep_rewards': ep_rewards
    }
    try:
        with open(f'checkpoints/{MODEL_NAME}_checkpoint.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)
        with open(f'replay/{MODEL_NAME}.pkl', 'wb') as f:
            pickle.dump(agent.replay_memory, f)
        agent.model.save(f'models/{MODEL_NAME}.keras')
        print(f"Checkpoint saved at episode {episode}.")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint(agent):
    try:
        model_path = f'models/{MODEL_NAME}.keras'
        replay_path = f'replay/{MODEL_NAME}.pkl'
        checkpoint_path = f'checkpoints/{MODEL_NAME}_checkpoint.pkl'

        if os.path.exists(model_path):
            agent.model = load_model(model_path)
            agent.target_model.set_weights(agent.model.get_weights())
            print(f"Loaded model from {model_path}")

        if os.path.exists(replay_path):
            with open(replay_path, 'rb') as f:
                agent.replay_memory = pickle.load(f)
                print(f"Loaded replay memory from {replay_path}")

        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            print(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    return {'episode': 1, 'progress_list': [], 'wins_list': [], 'ep_rewards': []}

def main():
    env = MinesweeperEnv(params.width, params.height, params.n_mines)
    agent = DQNAgent(env, MODEL_NAME)

    checkpoint = load_checkpoint(agent)
    start_episode = checkpoint['episode']
    progress_list = checkpoint['progress_list']
    wins_list = checkpoint['wins_list']
    ep_rewards = checkpoint['ep_rewards']

    for episode in tqdm(range(start_episode, params.episodes + 1), unit='episode'):
        agent.tensorboard.step = episode

        env.reset()
        episode_reward = 0
        past_n_wins = env.n_wins

        done = False
        while not done:
            current_state = env.state_im
            row, col = agent.get_action(current_state)
            new_state, reward, done = env.step(row, col)

            episode_reward += reward

            agent.update_replay_memory((current_state, (row, col), reward, new_state, done))
            agent.train(done)

        progress_list.append(env.n_progress)
        ep_rewards.append(episode_reward)

        if env.n_wins > past_n_wins:
            wins_list.append(1)
        else:
            wins_list.append(0)

        if not episode % AGG_STATS_EVERY:
            med_progress = round(np.median(progress_list[-AGG_STATS_EVERY:]), 2)
            win_rate = round(np.sum(wins_list[-AGG_STATS_EVERY:]) / AGG_STATS_EVERY, 2)
            med_reward = round(np.median(ep_rewards[-AGG_STATS_EVERY:]), 2)

            agent.tensorboard.update_stats(
                progress_med=med_progress,
                winrate=win_rate,
                reward_med=med_reward,
                learn_rate=agent.learn_rate,
                epsilon=agent.epsilon)

            print(f'Episode: {episode}, Median progress: {med_progress}, Median reward: {med_reward}, Win rate : {win_rate}')

        if not episode % SAVE_MODEL_EVERY:
            save_checkpoint(agent, episode, progress_list, wins_list, ep_rewards)

if __name__ == "__main__":
    main()
