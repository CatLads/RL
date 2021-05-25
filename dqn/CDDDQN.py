import tensorflow as tf
from dqn.DDDQN import DDDQN
from dqn.replay_buffer import ExpReplay
import numpy as np


class CDDDQN(DDDQN):
    def __init__(self, actions, input_shape):
        """Create a DQN network using convolutions.

        Args:
            actions (int): Number of actions
        """
        self.width, self.height = input_shape
        self.index = self.width*self.height
        super().__init__(actions)

    def network_setup(self):
        """
        Neural network structure treats splits input data into 2 chunks: one coming
        from TreeObservation and the other one coming from TemperatureObservation.
        Data from TreeObservation is processed by a dense layer.
        Data from TemperatureObservation is processed by 2 convolutional layers.
        Remaining layers are the same.
        """
        # temperature pipeline
        self.reshape = tf.keras.layers.Reshape((1, self.width, self.height))
        self.c1_temp = tf.keras.layers.Conv2D(32, 
                                              8,
                                              strides=(4, 4),
                                              padding="same",
                                              activation="relu")
        self.c2_temp = tf.keras.layers.Conv2D(64,
                                              4,
                                              strides=(2, 2),
                                              padding="same",
                                              activation="relu")
        self.flatten = tf.keras.layers.Flatten()
        # tree pipeline
        self.tree_input = tf.keras.layers.Dense(128, activation="relu")
        
        # merging spot
        self.concatenate = tf.keras.layers.Concatenate()
        self.d1 = tf.keras.layers.Dense(256, activation="relu")
        self.d2 = tf.keras.layers.Dense(128, activation="relu")
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(self.actions, activation=None)

    def _process_input(self, input_data):
        """
        Process the input data up to the advantage and action value.

        Args:
            input_data (tf.Tensor): Observation in input
        
        Returns:
            tf.Tensor: Preprocessed input
        """
        # split data of TreeObservation from the one on TemperatureObservation
        temperature_observation, tree_observation = tf.split(input_data, 
                                                             [self.index, (input_data.shape[1] - self.index)], 
                                                             axis=1)
        temp_x = self.reshape(temperature_observation)
        temp_x = self.c1_temp(temp_x)
        temp_x = self.c2_temp(temp_x)
        temp_x = self.flatten(temp_x)

        tree_x = self.tree_input(tree_observation)
        return self.concatenate([temp_x, tree_x])

    def call(self, input_data):
        """Forward pass.

                |-> [reshape] -> [conv1] -> [conv2] -> [flatten] -| |->[V]-|
        [data] -|                                                 |-|      |->[V + A - mean(A)]
                |-> [dense layer] --------------------------------| |->[A]-|

        Args:
            input_data (tf.Tensor): State observation

        Returns:
            tf.Tensor: Q-value estimate for each action
        """
        x = self._process_input(input_data)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return Q

    def advantage(self, input_data):
        """Get the action estimate given the state

        Args:
            state (np.array): State observation

        Returns:
            np.array: Estimate advantage per each action
        """
        x = self._process_input(input_data)
        x = self.d2(x)
        a = self.a(x)
        return a