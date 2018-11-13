import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import statistics
from statistics import mean, median
from collections import Counter

# learning rate
LR = 1E-3
# getting the environment from http://gym.openai.com/envs/CartPole-v1/
env = gym.make('CartPole-v1')
env.reset()
# want to hold the pole up for 500 frames
goal_steps = 500
score_requirement = 50
initial_games = 1000


def init_rand_games():
    ''' Renders the game for 5 games using random actions '''
    for _ in range(5):
        env.reset()
        for _ in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            if done:
                break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            obvservation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = obvservation
            score += reward

            if done:
                break

        if score > score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved_data.npy', training_data_save)
    print('average accepted score:', sum(
        accepted_scores) / len(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data


def new_population(model):
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []

        for _ in range(goal_steps):
            if len(prev_observation) == 0:
                action = random.randrange(0, 2)
            else:
                game_memory.append([prev_observation, action])
                action = np.argmax(
                    model.predict(
                        np.array(prev_observation).reshape(
                            -1, len(prev_observation)))[0])

            obvservation, reward, done, info = env.step(action)
            prev_observation = obvservation
            score += reward

            if done:
                break

        if score > 300:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] >= 0.5:
                    output = [0, 1]
                elif data[1] < 0.5:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved_data.npy', training_data_save)

    try:
        print('average accepted score:', mean(accepted_scores))
        print('Median score for accepted scores:', median(accepted_scores))
    except statistics.StatisticsError:
        print('no accepted scores, unable to print out stats')

    print(Counter(accepted_scores))
    return training_data


def neural_net_model(input_size):
    network = input_data(shape=[None, input_size], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(
        network, optimizer='adam', learning_rate=LR,
        loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')
    return model


def train_model(training_data, model=False):
    x = np.array([i[0] for i in training_data]
                 ).reshape(-1, len(training_data[0][0]))
    y = [i[1] for i in training_data]

    if not model:
        model = neural_net_model(input_size=len(x[0]))

    model.fit(
        {'input': x},
        {'targets': y},
        n_epoch=5, snapshot_step=500, show_metric=True,
        run_id='openai')
    return model


training_data = initial_population()
model1 = train_model(training_data)
# training_data = new_population(model1)
# model2 = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_observation = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_observation) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(
                model1.predict(
                    np.array(prev_observation).reshape(
                        -1, len(prev_observation)))[0])

        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_observation = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)

print('average score:', sum(scores) / len(scores))
print(
    'choice 1: {}, choice 0: {}'.format(
        choices.count(1) / len(choices),
        choices.count(0) / len(choices)))

model1.save('firstgen.model')
