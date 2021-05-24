import tensorflow as tf
from dqn.replay_buffer import ExpReplay
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class CDDQN(tf.keras.Model):
    """Implementation based on https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a"""

    def __init__(self, actions, batch_size, input_shape):
        """Create DQN network.

        Args:
            actions (int): Number of actions
        """
        super(CDDQN, self).__init__()
        self.actions = actions
        self.width, self.height = input_shape
        self.index = self.width*self.height
        # TODO: Take a look here, they used LTSM to solve some known problems
        #       https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
        # TODO: Note that we should handle things a little differently, its probably better to exploit Convolutions
        #       and also give to the network more than one observation at time. For the atari environment (the one used
        #       to highlight the power of DQN) they gave to the network multiple frames so that the network could learn
        #       about agent movements
        self.data_input_shape = (1, *input_shape)
        self.reshape = tf.keras.layers.Reshape(self.data_input_shape)
        self.c1 = tf.keras.layers.Conv2D(32,
                                         8,
                                         strides=(4, 4),
                                         padding="same",
                                         activation="relu",
                                         #input_shape=self.data_input_shape,
                                         )
        self.c2 = tf.keras.layers.Conv2D(64,
                                         4,
                                         strides=(2, 2),
                                         padding="same",
                                         activation="relu",
                                         )
        self.flatten = tf.keras.layers.Flatten()
        self.tree_input = tf.keras.layers.Dense(
            128, activation="relu"
        )
        self.tree_concatenate = tf.keras.layers.Concatenate()
        self.d1 = tf.keras.layers.Dense(256, activation="relu")
        self.d2 = tf.keras.layers.Dense(128, activation="relu")
        # v is used to estimate the value of a given state
        self.v = tf.keras.layers.Dense(1, activation=None)

        # a is used to compute the advantage of taking each action on a given state
        self.a = tf.keras.layers.Dense(self.actions, activation=None)

    def call(self, input_data):
        """Forward pass.
        Input data is initially passed through 2 dense layers and then splitted
        between V (estimate on how good the state is) and A (estimate on the advantage given by each action).
        Output of V and A are aggregated as
        Q = V + A - mean(a)

        Args:
            input_data (np.array): State observation

        Returns:
            np.array: Q-value estimate for each action
        """
        temperature_observation, tree_observation = tf.split(input_data, [self.index, (input_data.shape[1]-self.index)], axis=1)
        x = self.reshape(temperature_observation)
        x = self.c1(x)
        x = self.c2(x)
        x = self.flatten(x)
        tree_input = self.tree_input(tree_observation)
        concatenate = self.tree_concatenate([x, tree_input])
        x = self.d1(concatenate)
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
        temperature_observation, tree_observation = tf.split(input_data, [self.index, (input_data.shape[1]-self.index)], axis=1)
        x = self.reshape(temperature_observation)
        x = self.c1(x)
        x = self.c2(x)
        x = self.flatten(x)
        tree_input = self.tree_input(tree_observation)
        concatenate = self.tree_concatenate([x, tree_input])
        x = self.d1(concatenate)
        x = self.d2(x)
        a = self.a(x)
        return a


class Agent():
    """Based on https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a"""

    def __init__(self, observation_shape, actions, grid_shape, gamma=0.99, replace=3, lr=0.001, epsilon_decay=1e-3):
        """Create a DDQN agent.

        Args:
            observation_shape (tuple): Observation array shape
            actions (int): Number of actions
            gamma (float, optional): Discount factor in q-values update. Describes how much the
                                     previous q-values are going to be mantained. Defaults to 0.99.
            replace (int, optional): Number of iteration after which the target network weights are
                                     updated. Defaults to 100.
            lr (float, optional): Learning rate. Defaults to 0.001.
            epsilon_decay (float, optional): How much epsilon is decreased at each iteration.
                                             Defaults to 1e-3.
        """
        self.observation_shape = observation_shape
        self.actions = actions

        self.gamma = gamma
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 1 - 1e-5
        self.replace = replace

        self.trainstep = 0


        self.batch_size = 64
        self.memory = ExpReplay(observation_shape, self.batch_size)

        self.loss = np.zeros((self.batch_size))

        # Deep network creation
        self.q_net = CDDQN(self.actions, self.batch_size, grid_shape)
        self.target_net = CDDQN(self.actions, self.batch_size, grid_shape)
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss='mse', optimizer=opt)
        self.target_net.compile(loss='mse', optimizer=opt)

    def act(self, state):
        """Returns the action which is going to be taken using epsilon-greedy algorithm
        e.g. with probability epsilon choose a random action from the memory otherwise exploit
        the q-table.

        Args:
            state (np.array): State observation

        Returns:
            int: Best action
        """
        action = None

        if np.random.rand() <= self.epsilon:
            action = np.random.choice(list(range(self.actions)))
        else:
            actions = self.q_net.advantage(np.array([state]))
            action = np.argmax(actions)
        return action

    def update_mem(self, state, action, reward, next_state, done):
        """Add a sample to the experience replay

        Args:
            state (np.array): State observation
            action (np.array): Action taken
            reward (np.array): Reward after using action
            next_state (np.array): State obtained after executing action
            done (bool): True if state is final
        """
        self.memory.add(state, action, reward, next_state, done)

    def update_target(self):
        """Update target network (Double Q-Network)"""
        self.target_net.set_weights(self.q_net.get_weights())

    def update_epsilon(self):
        """Reduce epsilon by `epsilon_decay` value.

        Returns:
            np.float: Epsilon value
        """
        self.epsilon = self.epsilon * \
                       self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
        return self.epsilon

    def train(self):
        """Train the networks"""
        try:
            # update target network if needed
            if (self.trainstep % self.replace == 0) and self.trainstep>0:
                self.update_target()

            states, actions, rewards, next_states, dones, batch = self.memory.sample(
                self.batch_size)

            # get q-values estimate for each action
            target = self.q_net.predict(states)
            next_state_val = self.target_net.predict(
                next_states)  # value of the next state
            max_action = np.argmax(self.q_net.predict(
                next_states), axis=1)  # best action for next state
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # update estimates of q-values based on next state estimate
            # TODO: Why do we need the `* dones` bit? Do we actually need it?
            q_target = np.copy(target)
            q_target[batch_index, actions] = rewards + self.gamma * \
                next_state_val[batch_index, max_action] * dones

            # TEchnically, q_target is our real value, while target is the predicted one. So, target-q_target should be a good estimate of loss.
            # Is this loss?
            self.memory.losses[batch] = np.sum(
                np.abs(target - q_target), axis=1)
            # train the network
            self.q_net.train_on_batch(states, q_target)

            self.update_epsilon()
            self.trainstep += 1
        except MemoryError:
            # not enough samples in memory, wait to train
            pass

    def save_model(self):
        """Save weights locally"""
        self.q_net.save_weights("CDDQN_model")
        self.target_net.save_weights("CDDQN_target_model")
        with open("epsilon", "w") as f:
            f.write(str(self.epsilon))
        print("model saved")

    def load_model(self):
        """Load local weights"""
        self.q_net.load_weights("CDDQN_model")
        self.target_net.load_weights("CDDQN_target_model")
        try:
            with open("epsilon", "r") as f:
                self.epsilon = float(f.read().strip())
        except FileNotFoundError:
            print("epsilon file not found")
        print("model loaded")
