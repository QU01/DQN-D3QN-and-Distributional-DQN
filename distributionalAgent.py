import tensorflow as tf
from tensorflow import keras
import numpy as np

class DistributionalAgent:
    
    def __init__(self, env, dqn, target_dqn, epsilon, replay_buffer, support, lim):
        
        self.env = env
        self.network = dqn
        self.target_net = target_dqn
        self.epsilon = epsilon
        self.experience_replay = replay_buffer
        self.counter = 0
        self.update_rate = 80
        self.discount_factor = 0.9
        self.support = support
        self.lim = lim

        
    def sample_action(self, state):
        
        q_values = self.network(tf.expand_dims(state, axis=0))
        
        if np.random.random() > self.epsilon:
            
            preds = np.squeeze(self.network(tf.expand_dims(state, axis=0)).numpy())

            expectations = []

            for action in preds:

              expectations.append(self.support @ action)

            action = np.argmax(expectations)
            
        else:
            
            action = np.random.choice([i for i in range(self.env.action_space.n)])
            
        return action

    def get_actions(self, q_values):
        
        actions = []

        for q in q_values:


          expectations = []

          for dist in q:
            
            expectations.append(np.dot(self.support,dist))

          actions.append(np.argmax(expectations))

        return actions

    def get_target_dist(self, dist_batch, action_batch,reward_batch):

      nsup = self.support.shape[0]
      vmin,vmax = self.lim[0],self.lim[1]
      dz = (vmax-vmin)/(nsup-1.)
      target_dist_batch = dist_batch.numpy()

      for i in range(len(action_batch)):

          action = action_batch[i]
          dist = target_dist_batch[i,action,:]
          r = reward_batch[i]

          target_dist = self.update_dist(r,self.support,dist)

          target_dist_batch[i,action,:] = target_dist
          
      return target_dist_batch

    def update_dist(self, r,support,probs):

      nsup = probs.shape[0]
      vmin,vmax = self.lim[0],self.lim[1]
      dz = (vmax-vmin)/(nsup-1.)
      bj = np.round((r-vmin)/dz)
      bj = int(np.clip(bj,0,nsup-1))
      m = tf.identity(probs).numpy()
      j = 1
      
      for i in range(bj,1,-1):
          m[i] += np.power(self.discount_factor,j) * m[i-1]
          j += 1
      j = 1
      for i in range(bj,nsup-1,1): 
          m[i] += np.power(self.discount_factor,j) * m[i+1]
          j += 1
      m /= m.sum()
      return m
        

    def train(self):

      if len(self.experience_replay.buffer) < self.experience_replay.batch_size:

        return

      states, next_states, actions, rewards, dones = self.experience_replay.sample()

      q_values_next = self.target_net(next_states)
      actions_n = self.get_actions(q_values_next)
        
      target_dist = self.get_target_dist(q_values_next,actions_n, rewards)

      self.network.train_on_batch(states, target_dist)

      self.counter += 1

      if self.counter % self.update_rate == 0:

        self.target_net.set_weights(self.network.get_weights())
