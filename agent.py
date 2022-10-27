import tensorflow as tf
from tensorflow import keras
import numpy as np

class Agent:
    
    def __init__(self, env, dqn, target_dqn, epsilon, replay_buffer, double, noisy = False):
        
        self.env = env
        self.network = dqn
        self.target_net = target_dqn
        self.epsilon = epsilon
        self.experience_replay = replay_buffer
        self.double = double
        self.counter = 0
        self.update_rate = 80
        self.discount_factor = 0.9
        self.noisy = noisy

        
    def sample_action(self, state):
        
        q_values = self.network(tf.expand_dims(state, axis=0))
        
        if np.random.random() > self.epsilon:
            
            action = np.argmax(q_values)
            
        else:
            
            action = np.random.choice([i for i in range(self.env.action_space.n)])
            
        return action
        

    def train(self):

      if len(self.experience_replay.buffer) < self.experience_replay.batch_size:

        return

      states, next_states, actions, rewards, dones = self.experience_replay.sample()

      q_values = self.network(states).numpy()

      if not self.double:

        q_values_next = self.target_net(next_states)
        q_target = tf.math.reduce_max(q_values_next, axis=1, keepdims=True).numpy()

      else:
        q_values_next = self.network(next_states).numpy()
        q_values_next_tar = self.network(next_states)
        actions_n = tf.argmax(q_values_next_tar, axis=1).numpy()
        
        q_target = [q_vals[action] for q_vals, action in zip(q_values_next, actions_n)]

      q_pred = np.copy(q_values)

      assert len(q_target) == len(dones)

      for idx in range(len(q_target)):

        if not dones[idx]:

          q_pred[idx, actions[idx]] = q_target[idx] + rewards[idx]

        else:

          q_pred[idx, actions[idx]] = rewards[idx]


      self.network.train_on_batch(states, q_pred)

      self.counter += 1

      if self.counter % self.update_rate == 0:

        self.target_net.set_weights(self.network.get_weights())