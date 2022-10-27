import tensorflow as tf
from tensorflow import keras
import numpy as np

class CategoricalDQN(keras.models.Model):
    
    def __init__(self, num_actions, atoms, noisy = False, stdv = 0.9, atari = atari):
        super().__init__()
        
        self.num_actions = num_actions
        self.noisy = noisy
        self.stdv = stdv
        self.atari = atari
        self.atoms = atoms

      
        #Convolutional Layers

        if atari:
          self.conv1 = keras.layers.Conv2D(32, kernel_size = (4,4), strides=(2,2), padding="same", activation="relu")
          self.batchn1 = keras.layers.BatchNormalization()
          self.conv2 = keras.layers.Conv2D(64, kernel_size = (4,4), strides=(2,2), padding="same", activation="relu")
          self.batchn2 = keras.layers.BatchNormalization()
          self.conv3 = keras.layers.Conv2D(128, kernel_size = (4,4), strides=(2,2), padding="same", activation="relu")
          self.batchn3 = keras.layers.BatchNormalization()
          self.conv4 = keras.layers.Conv2D(256, kernel_size = (4,4), strides=(2,2), padding="same", activation="relu")
          self.batchn4 = keras.layers.BatchNormalization()
          self.flatten = keras.layers.Flatten()
        
        #Fully Connected Layers
        self.hidden = keras.layers.Dense(256, activation = "relu")
        self.hidden1 = keras.layers.Dense(256, activation = "relu")
        self.hidden2 = keras.layers.Dense(128, activation = "relu")
        self.outputs = []

        for i in range(self.num_actions):
          self.outputs.append(keras.layers.Dense(atoms, activation="softmax"))

        #Noise Layers
        if self.noisy:
          self.noise = keras.layers.GaussianNoise(self.stdv, seed=741)
        
    def call(self, inputs):

        if self.atari: 
          x = self.conv1(inputs)
          x = self.batchn1(x)

          x = self.conv2(x)
          x = self.batchn2(x)

          x = self.conv3(x)
          x = self.batchn3(x)

          x = self.conv4(x)
          x = self.batchn4(x)

          inputs = self.flatten(x)
        
        h = self.hidden(inputs)
        h = self.hidden1(h)
        h = self.hidden2(h)



        if self.noisy:

          h = self.noise(h)


        outputs = []

        for output in self.outputs:

          q_values = output(h)
          outputs.append(q_values)
               
        return tf.reshape(tf.concat(outputs, axis=0), [inputs.shape[0], self.num_actions, self.atoms])