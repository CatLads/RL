from configparser import ConfigParser
import tensorflow as tf
from dqn.replay_buffer import ExpReplay
from dqn.CDDDQN import CDDDQN
from dqn.DDDQN import DDDQN
import numpy as np

config_object = ConfigParser()


class Agent():
    """Based on https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a"""

    def __init__(self, observation_shape, actions, grid_shape, gamma=0.99, replace=100, lr=0.001,
                 epsilon_decay=1e-3, decay_type="flat", initial_epsilon=1.0, min_epsilon=0.01,
                 batch_size=64, method="DDDQN"):
        """Create a DDQN agent.

        Args:
            observation_shape (tuple): Observation array shape
            actions (int): Number of actions
            grid_shape (tuple): Shape of the train grid
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
            method (str, optional): The algorithm to use. Either DDDQN or CDDDQN. Defaults to DDDQN.
        """
        self.observation_shape = observation_shape
        self.memory = ExpReplay(observation_shape)
        self.actions = actions
        self.grid_shape = grid_shape

        # hyperparameters input
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.replace = replace
        self.batch_size = batch_size
        self.decay_type = decay_type.lower() if decay_type.lower() in [
            "flat", "smooth"] else "flat"
        self.method = method.lower() if method.lower() in [
            "dddqn", "cdddqn"] else "dddqn"

        self.trainstep = 0

        # Deep network creation
        if self.method == "dddqn":
            self.q_net = DDDQN(self.actions)
            self.target_net = DDDQN(self.actions)
        elif self.method == "cdddqn":
            self.q_net = CDDDQN(self.actions, self.grid_shape)
            self.target_net = CDDDQN(self.actions, self.grid_shape)

        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        self.q_net.compile(loss="mse", optimizer=opt)
        self.target_net.compile(loss="mse", optimizer=opt)

    def act(self, state, legal_moves):
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
            # exploit numpy's probability feature to select only legal moves
            probabilities = legal_moves/(np.sum(legal_moves))
            action = np.random.choice(list(range(self.actions)),
                                      p=probabilities)
        else:
            # keras model expects at least 2D data
            actions = self.q_net.advantage(np.array([state]))
            # extract actions and mask illegal action by setting the q-value low
            actions = actions.numpy()[0]
            actions[legal_moves == 0] = -1e4
            # select the best action
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
           
            # build indexes for all predicted q-values
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            # update estimates of q-values on the targetnet based on qnet estimation
            q_target = np.copy(target)
            q_target[batch_index, actions] = rewards + self.gamma * \
                next_state_val[batch_index, max_action] * dones

            # train network and set losses on the experience replay
            self.memory.losses[batch] = self.q_net.train_on_batch(
                states, q_target)

            self.update_epsilon()
            self.trainstep += 1
        except MemoryError:
            # not enough samples in memory, wait to train
            pass

    def save_model(self):
        """Save the network models"""
        self.q_net.save_weights("%s_qnet_model" % self.method)
        self.target_net.save_weights("%s_targetnet_model" % self.method)

        # save hyperparameters
        config_object["CONFIG"] = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay": self.epsilon_decay,
            "replace": self.replace,
            "batch_size": self.batch_size,
            "decay_type": self.decay_type,
            "method": self.method
        }
        with open("%s_config.ini" % self.method, "w") as f:
            config_object.write(f)

        print("model saved")

    def load_model(self):
        """Load local weights"""
        self.q_net.load_weights("%s_qnet_model" % self.method)
        self.target_net.load_weights("%s_targetnet_model" % self.method)

        # load hyperparameters
        config = config_object.read("%s_config.ini" % self.method)["CONFIG"]
        self.gamma = config["gamma"]
        self.min_epsilon = config["min_epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.replace = config["replace"]
        self.batch_size = config["batch_size"]
        self.decay_type = config["decay_type"]
        self.method = config["method"]

        print("model loaded")
