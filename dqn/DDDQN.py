from configparser import ConfigParser
import tensorflow as tf
from dqn.replay_buffer import ExpReplay
import numpy as np

config_object = ConfigParser()

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
        x = self.d1(input_data)
        x = self.d2(x)
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
        x = self.d1(state)
        x = self.d2(x)
        a = self.a(x)
        return a


class Agent():
    """Based on https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a"""

    def __init__(self, observation_shape, actions, gamma=0.99, replace=100, lr=0.001, epsilon_decay=1e-3,
                 decay_type="flat", initial_epsilon=1.0, min_epsilon=0.01, batch_size=64):
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
            decay_type (str, optional): Type of decay implemented, can be either flat, smooth.
                                        Defaults to flat.
            intial_epsilon (float, optional): Defaults to 1.0.
            min_epsilon (float, optional): Defaults to 0.01.
            batch_size (int, optional): Defaults to 64.
        """
        self.observation_shape = observation_shape
        self.memory = ExpReplay(observation_shape)
        self.actions = actions

        # hyperparameters input
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.replace = replace
        self.batch_size = batch_size
        self.decay_type = decay_type if decay_type in ["flat", "smooth"] else "flat"

        self.trainstep = 0

        # Deep network creation
        self.q_net = DDDQN(self.actions)
        self.target_net = DDDQN(self.actions)
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
            # FIXME: Why np.array([state])?
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
        decay_fn = {
            "flat": lambda e: e - self.epsilon_decay,
            "smooth": lambda e: e * (1 - self.epsilon_decay)
        }

        if self.epsilon > self.min_epsilon:
            self.epsilon = decay_fn[self.decay_type](self.epsilon)  
        else:
            self.epsilon = self.min_epsilon
        
        return self.epsilon

    def train(self):
        """Train the networks"""
        try:
            # update target every replace iterations
            if self.trainstep > 0 and self.trainstep % self.replace == 0:
                self.update_target()

            states, actions, rewards, next_states, dones, batch = self.memory.sample(
                self.batch_size)

            # get q-values estimate for each action
            target = self.q_net.predict(states)
            # value of the next state
            next_state_val = self.target_net.predict(next_states)
            # best action for next state
            max_action = np.argmax(self.q_net.predict(next_states), axis=1)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # update estimates of q-values based on next state estimate
            # FIXME: Why do we need the `* dones` bit? Do we actually need it?
            q_target = np.copy(target)
            q_target[batch_index, actions] = rewards + self.gamma * \
                                             next_state_val[batch_index, max_action] * dones

            self.memory.losses[batch] = np.sum(np.abs(target - q_target), axis=1)
            # train the network
            self.q_net.train_on_batch(states, q_target)

            self.update_epsilon()
            self.trainstep += 1
        except MemoryError:
            # not enough samples in memory, wait to train
            pass

    def save_model(self, name="DDDQN"):
        """Save the network models

        Args:
            name (str, optional): Prefix to each model. Defaults to "DDDQN".
        """
        self.q_net.save("%s_qnet_model" % name)
        self.target_net.save("%s_targetnet_model" % name)

        # save hyperparameters
        config_object["CONFIG"] = {
            "gamma": self.gamma
            "epsilon": self.initial_epsilon
            "min_epsilon": self.min_epsilon
            "epsilon_decay": self.epsilon_decay
            "replace": self.replace
            "batch_size": self.batch_size
            "decay_type": self.decay_type
        }
        with open("%s_config.ini" % name, "w") as f:
            config_object.write(f)

        print("model saved")

    def load_model(self, name="DDDQN"):
        """Load local weights"""
        self.q_net = tf.keras.models.load_model("%s_qnet_model" % name)
        self.target_net = tf.keras.models.load_model("%s_targetnet_model" % name)
        
        # load hyperparameters
        config = config_object.read("%s_config.ini" % name)["CONFIG"]
        self.gamma = config["gamma"]
        self.epsilon = config["initial_epsilon"]
        self.min_epsilon = config["min_epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.replace = config["replace"]
        self.batch_size = config["batch_size"]
        self.decay_type = config["decay_type"]
        
        print("model loaded")
