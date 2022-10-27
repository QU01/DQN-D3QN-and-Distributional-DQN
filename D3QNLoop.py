import tensorflow as tf
import gym
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import datetime, os
from dqn import DQN
import pandas as pd
from agent import Agent
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

    main_net = DQN(env.env.action_space.n, dueling=True, noisy=False)

    main_net.compile(optimizer=tf.keras.optimizers.Adam(0.00075), loss="mse")
    main_net(tf.expand_dims(env.reset()[0], axis=0))

    main_net.summary()

    target_net = DQN(env.env.action_space.n, dueling=True, noisy=False)
    target_net(tf.expand_dims(env.reset()[0], axis=0))

    target_net.set_weights(main_net.get_weights())

    experience_replay = ReplayBuffer(1000,64)

    agent = Agent(env, main_net, target_net, 1, experience_replay, True)

    epochs = 200

    min_epsilon = 0.05

    epsilon_decay = (agent.epsilon-min_epsilon)/100

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

            action = agent.sample_action(state)
            next_state, reward, done, _, info = env.step(action)
            Return += reward
            agent.experience_replay.append(state, next_state, action, reward, done, info)
            state = next_state
            steps += 1

        agent.train()

        returns.append(Return)
        avg_return = np.mean(returns[-10:])
        avg_returns.append(avg_return)

        epsilons.append(agent.epsilon)
        agent.epsilon = np.max([agent.epsilon-epsilon_decay, min_epsilon])

        agent.target_net.set_weights(agent.network.get_weights())

        agent.network.save(f"dqn_experiment_{ epoch }", save_format='tf')

        total_steps += steps
        
        print("Episode: " + str(epoch)+"/"+str(epochs) + " return of "+ str(Return) + " average reward of: " + str(avg_return) + " epsilon: " + str(agent.epsilon) + "in " + str(steps) + " steps")

        conclusions_experiment1 = pd.DataFrame({"Returns": returns, "Average Return (Last 10 episodes)": avg_return, "Epsilon": epsilons})

        conclusions_experiment1.to_csv("conclusions_experiment1.csv")

        agent.network.save("dqn_experiment_1", save_format='tf')
    