import argparse
import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

from DDGP import DDPG


def train_agent(n_episodes=200, add_noise=True):
    n_episodes_avg = 100
    scores_deque = deque(maxlen=n_episodes_avg)
    ep_scores = []
    avg_scores = []
    steps_in_episode = 1000
    scores_filename, start_episode = load_scores(avg_scores, ep_scores, scores_deque)

    for n_episode in range(start_episode, start_episode + n_episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        for episode in range(1, steps_in_episode + 1):
            actions = []
            for i, agent in enumerate(agents):
                actions.append(agent.act(states[i], add_noise))
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for i, agent in enumerate(agents):
                agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])

            for i in range(len(rewards)):
                scores[i] += rewards[i]
            states = next_states
            if np.any(dones):
                break

        for agent in agents:
            agent.reset_noise()
        episode_score = np.max(scores)
        scores_deque.append(episode_score)
        average_score = np.mean(scores_deque)
        ep_scores.append(episode_score)
        avg_scores.append(average_score)
        with open(scores_filename, 'a') as f:
            f.write(f'{episode_score}\n')

        print(f'Episode {n_episode} score: {episode_score:.3f}, average score: {average_score:.3f}')
        for i, agent in enumerate(agents):
            agent.save_weights(i+1)
        if n_episode >= 100 and average_score >= 2.0:
            print(f'Average score goal reached! Training takes: {n_episode - 100} episodes')
            break

    return ep_scores, avg_scores


def load_scores(avg_scores, ep_scores, scores_deque):
    start_episode = 0
    scores_filename = f'./weights/{args.checkpoint_prefix}episode_scores.txt'
    if os.path.isfile(scores_filename):
        with open(scores_filename, 'r') as f:
            for score_str in f:
                score = float(score_str[:-2])
                start_episode += 1
                scores_deque.append(score)
                average_score = np.mean(scores_deque)
                ep_scores.append(score)
                avg_scores.append(average_score)
    return scores_filename, start_episode


def test_agent(n_episodes, add_noise=False):
    for n_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            actions = agents[0].act(states, add_noise)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            states = next_states
            if np.any(dones):
                break
        episode_score = np.mean(scores)
        print(f'Episode {n_episode} score: {episode_score :.3f}')


def plot(ep_scores, avg_scores):
    plt.plot(ep_scores, label='Episode Score', marker='o', markersize=2, color='red', linewidth=1)
    plt.plot(avg_scores, label='Average Score', marker='o', markersize=2, color='blue', linewidth=1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', help='number of episodes', default=200, type=int)
    parser.add_argument('--port', help='application port', default=8888, type=int)
    parser.add_argument('--checkpoint_prefix', help='prefix of checkpoint filename', default='')
    parser.add_argument('--load_checkpoint', help='enable loading checkpoint', dest='load_checkpoint',
                        action='store_true')
    parser.add_argument('--test_mode', help='enable test mode', dest='test_mode', action='store_true')
    parser.add_argument('--no_noise', help='disable noise', dest='no_noise', action='store_true')
    args = parser.parse_args()

    training_mode = not args.test_mode
    noise = not args.no_noise
    env = UnityEnvironment('./Tennis_Windows_x86_64/Tennis.exe', no_graphics=training_mode, base_port=args.port)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=training_mode)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    agents = [DDPG(nS=state_size, nA=action_size, random_seed=42, load_checkpoint=args.load_checkpoint,
                          checkpoint_prefix=args.checkpoint_prefix, agent_no=1),
              DDPG(nS=state_size, nA=action_size, random_seed=42, load_checkpoint=args.load_checkpoint,
                          checkpoint_prefix=args.checkpoint_prefix, agent_no=2)]
    if args.test_mode:
        test_agent(n_episodes=args.n_episodes, add_noise=noise)
    else:
        episode_scores, average_scores = train_agent(n_episodes=args.n_episodes, add_noise=noise)
        plot(episode_scores, average_scores)
