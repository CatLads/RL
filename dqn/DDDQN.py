import tensorflow as tf
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from configparser import ConfigParser

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class DDDQN(tf.keras.Model):
    """Implementation based on https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a"""

    def __init__(self, actions):
        """Creates DQN network.

        Args:
            actions (int): Number of actions
        """
        super(DDDQN, self).__init__()
        self.actions = actions
        # TODO: Take a look here, they used LTSM to solve some known problems
        #       https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
        # TODO: Note that we should handle things a little differently, its probably better to exploit Convolutions
        #       and also give to the network more than one observation at time. For the atari environment (the one used
        #       to highlight the power of DQN) they gave to the network multiple frames so that the network could learn
        #       about agent movements
        self.network_setup()

    def network_setup(self):
        """
        Neural network structure is composed of 2 dense layers used create an internal representation of the observation.
        Output of the dense layers is then fed in parallel to 2 different dense layers: V and A.
        V is used to estimate how good is the observation.
        A is used to estimate how good each action seems to be from the current state.        
        """
        self.d1 = tf.keras.layers.Dense(128, activation="relu")
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
        x = self.d1(input_data)
        x = self.d2(x)
        return x

    def call(self, input_data):
        """Forward pass.
        Input data is initially passed through 2 dense layers and then splitted
        between V and A.
        Output of V and A are aggregated as 
           Q = V + A - mean(a)

                                  |---> [V] --|
        [data] -> [d1] -> [d2] ---|           |---->[V + A - mean(A)]
                                  |---> [A] --|

        Args:
            input_data (tf.Tensor): State observation

        Returns:
            tf.Tensor: Q-value estimate for each action
        """
        x = self._process_input(input_data)
        v = self.v(x)
        a = self.a(x)
        q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return q

    def advantage(self, state):
        """Get the action estimate given the state

        Args:
            state (tf.Tensor): State observation

        Returns:
            tf.Tensor: Estimate advantage for each action
        """
        x = self._process_input(input_data)
        a = self.a(x)
        return a
