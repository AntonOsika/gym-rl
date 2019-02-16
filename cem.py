import numpy as np
import pandas as pd
import gym
import matplotlib.pyplot as plt
import matplotlib as mpl

N_EPISODES = 100


class CEM:
    """
    parameters theta have an indepenet set of theta for each action
    """

    def __init__(
            self,
            n_actions,
            obs_shape,
            elite_frac=0.2,
            pop_size=200,
    ):
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.theta = np.random.rand(*(
            (pop_size, ) + obs_shape + (n_actions, )))
        self.means = np.zeros(self.theta.shape[1:])
        self.best_theta = self.means

        self.n_elites = int(elite_frac * pop_size)
        self.pop_size = pop_size
        self.reward_buffer = []

    def buffer_filled(self):
        return self.pop_size == len(self.reward_buffer) 
    
    def q_values(self, obs, agent): 
        return obs @ self.theta[agent]

    def act(self, obs):
        qs = self.q_values(obs, len(self.reward_buffer))
        return qs.argmax()

    def act_greedy(self, obs):
        return (obs @ self.best_theta).argmax()

    def feedback(self, reward_sum):
        self.reward_buffer.append(reward_sum)

    def update(self):
        sorted_pop = list(
            sorted((reward, i) for i, reward in enumerate(self.reward_buffer)))
        elites = sorted_pop[-self.n_elites:]
        elite_idxs = [i for _, i in elites]
        elite_theta = self.theta[elite_idxs]

        self.best_theta = self.theta[elite_idxs[-1]]

        stds = elite_theta.std(axis=0)
        eps = np.random.rand(*self.theta.shape)
        noise = eps.reshape([self.pop_size, -1]) * stds.reshape([-1])

        self.means = elite_theta.mean(axis=0)
        self.theta = self.means + noise.reshape(self.theta.shape)
        self.reward_buffer = []

        print(self.means)

env = gym.make('LunarLander-v2')
verbose = 0

agent = CEM(env.action_space.n, env.observation_space.shape)
rewardss = []
for episode in range(N_EPISODES):
    while not agent.buffer_filled():
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

            obs1 = obs2
        agent.feedback(sum(rewards))
        rewardss.append(sum(rewards))
    agent.update()
    print(episode)

    plot_every = 1
    if (1 + episode) % plot_every == 0:
        smoothed_rewards = pd.Series(rewardss).ewm(com=5)
        xs = range(len(rewardss))
        means = smoothed_rewards.mean()
        stds = smoothed_rewards.std()
        plt.plot(xs, means, color='b')
        plt.fill_between(
            xs[-plot_every - 1:],
            means[-plot_every - 1:] - stds[-plot_every - 1:],
            means[-plot_every - 1:] + stds[-plot_every - 1:],
            alpha=0.1,
            facecolor='b')
        plt.pause(0.001)

        obs1 = env.reset()
        done = False
        rewards = []
        while not done:
            action = agent.act_greedy(obs1)
            obs2, reward, done, info = env.step(action)
            rewards.append(reward)

            # env.render()
            # print('{:6.1f},    '.format(reward), '#' * int((reward + 5)))

            obs1 = obs2
        print(sum(rewards))

input()
