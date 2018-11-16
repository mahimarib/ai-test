import numpy as np
import gym
import random
import os
import time

env = gym.make('Taxi-v2')
num_of_actions = env.action_space.n
num_of_states = env.observation_space.n
# amount of games we want it to play to test
total_episodes = 50000
# total amount of test games we want to use
total_test_episodes = 10
# the max amount of actions it can make in a game
max_steps = 99
learning_rate = 0.7
'''
Discount Rate
We define a discount rate called gamma. It must be between 0 and 1.

The larger the gamma, the smaller the discount. This means the learning agent
cares more about the long term reward. On the other hand, the smaller the
gamma, the bigger the discount. This means our agent cares more about
the short term reward.
'''
gamma = 0.618
'''
Exploration Rate
The higher the exploration rate (epsilon) the more random the actions are
in order to explore the environment, this is to find the bigger reward
rather than exploiting the known information to maximize the reward.
'''
epsilon = 1.0
# the maximum rate, this will be the rate in the beginning
max_epsilon = 1.0
# this will the minimum rate, which will be at the end
min_epsilon = 0.01
# exploration decay rate for epsilon, we want more exporation
# in the beginning to know the environment, then exploit later
# to maximize the reward
rate_decay = 0.01


def get_qtable(epsilon=epsilon):
    q_table = np.zeros((num_of_states, num_of_actions))
    for episode in range(total_episodes):
        state = env.reset()
        exploration_exploitation = random.uniform(0, 1)
        for step in range(max_steps):
            # if the tradeoff is bigger than the epsilon then it will do
            # the action with the greatest valu for the state
            if exploration_exploitation > epsilon:
                action = np.argmax(q_table[state, :])
            # if not it will choose randomly (this will happen for the
            # the very first time)
            else:
                action = env.action_space.sample()
            new_state, reward, done, _ = env.step(action)
            # updating the bellman equation
            # Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
            state = new_state
            if done:
                break
        # updating the epsilon because we don't need the decay rate to be
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * \
            np.exp(-rate_decay * episode)
        print(episode + 1, '/', total_episodes, end='\r')
    return q_table


def play(qtable):
    env.reset()
    rewards = []
    for episode in range(total_test_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0
        for step in range(max_steps):
            os.system('clear')
            env.render()
            time.sleep(0.2)
            # Take the action (index) that have the maximum expected future
            # reward given that state
            action = np.argmax(qtable[state, :])
            new_state, reward, done, _ = env.step(action)
            total_rewards += reward
            if done:
                rewards.append(total_rewards)
                # print ("Score", total_rewards)
                break
            state = new_state
        time.sleep(1)
    env.close()
    print("Score over time: " + str(sum(rewards) / total_test_episodes))


q_table = np.load('qtable.npy')
play(q_table)
