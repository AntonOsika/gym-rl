import numpy as np
import gym
from gym.spaces import Discrete, Box

# ================================================================
# Policies
# ================================================================


class DeterministicDiscreteActionLinearPolicy(object):
    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0:dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions:None].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a


class DeterministicContinuousActionLinearPolicy(object):
    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0:dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac:None]

    def act(self, ob):
        a = np.clip(
            ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a


def do_episode(policy, env, discount=1.0, render=False):
    disc_total_rew = 0
    ob = env.reset()
    done = False
    t = 0
    while not done:
        a = policy.act(ob)
        (ob, reward, done, _info) = env.step(a)
        disc_total_rew += reward * discount**t
        if render and t % 3 == 0:
            env.render()
        t += 1
    return disc_total_rew


env = None


def noisy_evaluation(theta, discount=0.90):
    policy = make_policy(theta)
    reward = do_episode(policy, env, discount)
    return reward


def make_policy(theta):
    if isinstance(env.action_space, Discrete):
        return DeterministicDiscreteActionLinearPolicy(
            theta, env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return DeterministicContinuousActionLinearPolicy(
            theta, env.observation_space, env.action_space)
    else:
        raise NotImplementedError


# Task settings:
env = gym.make('LunarLander-v2')

# Alg settings:
n_iter = 200  # number of iterations of CEM
batch_size = 25  # number of samples per batch
elite_frac = 0.2  # fraction of samples used as elite set
n_elite = int(batch_size * elite_frac)
extra_std = 2.0
extra_decay_time = 10

if isinstance(env.action_space, Discrete):
    dim_theta = (env.observation_space.shape[0] + 1) * env.action_space.n
elif isinstance(env.action_space, Box):
    dim_theta = (
        env.observation_space.shape[0] + 1) * env.action_space.shape[0]
else:
    raise NotImplementedError

# Initialize mean and standard deviation
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)

# Now, for the algorithm
for itr in range(n_iter):
    # Sample parameter vectors
    extra_cov = max(1.0 - itr / extra_decay_time, 0) * extra_std**2
    thetas = np.random.multivariate_normal(
        mean=theta_mean,
        cov=np.diag(np.array(theta_std**2) + extra_cov),
        size=batch_size)
    rewards = np.array([noisy_evaluation(theta) for theta in thetas])

    # Get elite parameters
    elite_inds = rewards.argsort()[-n_elite:]
    elite_thetas = thetas[elite_inds]

    # Update theta_mean, theta_std
    theta_mean = elite_thetas.mean(axis=0)
    theta_std = elite_thetas.std(axis=0)
    print("iteration %i. mean f: %8.3g. max f: %8.3g" % (itr, np.mean(rewards),
                                                         np.max(rewards)))
    if itr % 10 == 0:
        print(theta_mean)
        do_episode(make_policy(theta_mean), env, discount=0.90, render=True)

