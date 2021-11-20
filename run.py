from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
from itertools import product


# Run dqn with Tetris
def dqn(episodes=6000,
        max_steps=None,
        epsilon_stop_episode=2000,
        epsilon_min=0.01,
        mem_size=20000,
        discount=0.95,
        batch_size=512,
        epochs=1,
        render_every=1000,
        log_every=100,
        replay_start_size=2000,
        train_every=1,
        n_neurons=[32, 32],
        render_delay=None,
        activations=['relu', 'relu', 'linear']):
    env = Tetris()

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons,
                     activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode,
                     mem_size=mem_size,
                     epsilon_min=epsilon_min,
                     discount=discount,
                     replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-eps_min={epsilon_min}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        # for episode in range(episodes):
        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())

            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)

            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)

    hparams = {"discount": discount, "epsilon_min": epsilon_min,
               "epsilon_stop_episode": epsilon_stop_episode}
    metrics = {"avg_score": mean(scores[-log_every:]),
               "min_score": min(scores[-log_every:]),
               "max_score": max(scores[-log_every:])}

    log.hparams(hparams, metrics)


if __name__ == "__main__":
    discount_list = [0.95, 0.9, 0.85, 0.8]
    epsilon_min_list = [0.02, 0.04, 0.08, 0.2]
    epsilon_stop_episode_list = [1500, 3000, 4500, 6000]

    for discount, epsilon_min, epsilon_stop_episode in product(
            discount_list, epsilon_min_list, epsilon_stop_episode_list):
        print(discount, epsilon_min, epsilon_stop_episode)
        dqn(discount=discount,
            epsilon_min=epsilon_min,
            epsilon_stop_episode=epsilon_stop_episode,
            render_every=None)
