import gym
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import matplotlib as mpl

import pandas as pd


N_EPISODES = 100

class Agent:
    """
    parameters theta have an indepenet set of theta for each action
    """

    def __init__(
            self,
            n_actions,
            obs_shape,
            episodes=N_EPISODES,
            eps_start=1.0,
            eps_end=1e-3,
            lr_start=0.01,
            lr_end=0.001,
            gamma=0.95,
    ):
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.theta = np.zeros(obs_shape + (n_actions, ))

        self.episode_count = 1
        self.episodes = episodes
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.gamma = gamma

    def reset(self):
        self.episode_count += 1

    def exp_decayed(self, start, end, progress):
        return start * (end / start)**progress

    @property
    def eps(self):
        return self.exp_decayed(self.eps_start, self.eps_end,
                                self.episode_count / self.episodes)

    @property
    def lr(self):
        return self.exp_decayed(self.lr_start, self.lr_end,
                                self.episode_count / self.episodes)

    def phi(self, obs):
        return obs

    def q_values(self, obs):
        features = self.phi(obs)
        return np.dot(features, self.theta)

    def act(self, obs):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.n_actions)
        qs = self.q_values(obs)
        return qs.argmax()


#     def update(self, obs1, action, obs2, reward):
#         self._update(self.phi(obs1),...)

    def update(self, features1, action, features2, reward):
        q2 = features2.dot(self.theta).max()
        q1 = features1.dot(self.theta)[action]
        delta = reward + self.gamma * q2 - q1
        self.theta[:, action] += self.lr * delta * features1

env = gym.make('LunarLander-v2')

verbose = 0


def train(color, **kwargs):
    print("params: ", kwargs)
    agent = Agent(env.action_space.n, env.observation_space.shape, **kwargs) 
    rewardss = []
    for episode in range(N_EPISODES):
        obs1 = env.reset()
        done = False
        rewards = []
        while not done:
            action = agent.act(obs1)
            obs2, reward, done, info = env.step(action)
            rewards.append(reward)

            if verbose:
                env.render()
                print('{:6.1f},    '.format(reward), '#' * int((reward + 5)))

            agent.update(obs1, action, obs2, reward)
            obs1 = obs2

        if verbose:
            print()
            sleep(1)

        rewardss.append(sum(rewards))

        if (1 + episode) % 10 == 0:
            smoothed_rewards = pd.Series(rewardss).ewm(com=5)
            xs = range(len(rewardss))
            means = smoothed_rewards.mean()
            stds = smoothed_rewards.std()
            plt.plot(xs, means)
            plt.fill_between(
                    xs[-1:], means[-1:] - stds[-1:], means[-1:] + stds[-1:], alpha=0.5, facecolor=color)
            plt.pause(0.001)


def a1():
    eps_start = 1.0
    eps_end = 1e-2
    lr_start = 0.01
    lr_end = 0.001
    gamma = 0.95
    return locals()

def a2():
    eps_start = 1.0
    eps_end = 1e-3
    lr_start = 0.001
    lr_end = 0.0001
    gamma = 0.95
    return locals()

def a3():
    eps_start = 1.0
    eps_end = 1e-3
    lr_start = 0.1
    lr_end = 0.01
    gamma = 0.95
    return locals()

params = [a1(), a2(), a3()]
cm = mpl.cm.get_cmap('viridis', len(params))

for i in range(len(params)):
    train(cm.colors[i], **params[i])

input()
