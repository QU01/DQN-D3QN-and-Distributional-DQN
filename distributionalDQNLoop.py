import tensorflow as tf
import gym
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime, os
from categoricalDQN import CategoricalDQN
import pandas as pd
from distributionalAgent import DistributionalAgent
from replay_buffer import ReplayBuffer
from gym.wrappers import AtariPreprocessing

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

if __name__ == "__main__":

    atari = False

    env = gym.make("CartPole-v1")

    if atari:
        env = AtariPreprocessing(env, frame_skip=1, grayscale_newaxis=True)
        env = ScaledFloatFrame(env)
    
    support = np.linspace(-200,200,51)
    replay_buffer = ReplayBuffer(10000, 64)

    categorical_net = CategoricalDQN(env.action_space.n, 51, noisy=False)
    categorical_net.compile(optimizer=tf.keras.optimizers.Adam(0.00075), loss="mse")
    categorical_net(tf.expand_dims(env.reset()[0], axis=0))
    categorical_net.summary()

    target_net = CategoricalDQN(env.action_space.n, 51, noisy=False)
    target_net(tf.expand_dims(env.reset()[0], axis=0))

    target_net.set_weights(categorical_net.get_weights())

    dagent = DistributionalAgent(env, categorical_net, target_net, 0.9, replay_buffer, support, (-200,200))

    epochs = 350

    min_epsilon = 0.05

    epsilon_decay = (dagent.epsilon-min_epsilon)/150

    returns = []
    avg_returns = []
    epsilons = []
    total_steps = 0

    state = env.reset()[0]

    for epoch in range(epochs):
        

        Return = 0
        done = False
        state = env.reset()[0]
        steps = 0

        while not done:

            action = dagent.sample_action(state)
            next_state, reward, done, _, info = env.step(action)
            Return += reward
            dagent.experience_replay.append(state, next_state, action, reward, done, info)
            state = next_state
            steps += 1

        dagent.train()

        returns.append(Return)
        avg_return = np.mean(returns[-10:])
        avg_returns.append(avg_return)

        epsilons.append(dagent.epsilon)
        dagent.epsilon = np.max([dagent.epsilon-epsilon_decay, min_epsilon])

        dagent.target_net.set_weights(dagent.network.get_weights())

        total_steps += steps
        
        print("Episode: " + str(epoch)+"/"+str(epochs) + " return of "+ str(Return) + " average reward of: " + str(avg_return) + " epsilon: " + str(dagent.epsilon) + "in " + str(steps) + " steps")

    conclusions_experiment1 = pd.DataFrame({"Returns": returns, "Average Return (Last 10 episodes)": avg_return, "Epsilon": epsilons})

    conclusions_experiment1.to_csv("conclusions_experiment1.csv")

    agent.network.save("distdqn_experiment_1", save_format='tf')