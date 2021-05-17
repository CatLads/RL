import numpy as np


class ExpReplay():
    """Implementation based on https://towardsdatascience.com/dueling-double-deep-q-learning-using-tensorflow-2-x-7bbbcec06a2a"""

    def __init__(self, state_shape, buffer_size=1000000):
        """
        Replay buffer
        The buffer will contain tuples of the form (state, action, reward, next_state, done).

        Args:
            state_shape ([np.array]): Shape of each state
            buffer_size (int, optional): [description]. Defaults to 1000000.
        """
        self.buffer_size = buffer_size

        self.state_shape = state_shape
        self.alpha = 0.6
        self.epsilon = 0.01
        self.state_mem = np.zeros(
            (self.buffer_size, *self.state_shape), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size), dtype=np.int32)
        # SIMMY: I removed , *action_shape cause we should just save one action per agent, not all of them
        self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_state_mem = np.zeros(
            (self.buffer_size, *state_shape), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
        self.losses = np.zeros((self.buffer_size), dtype=np.float32)

        # pointer is used to access the memory
        self.pointer = 0

    def add(self, state, action, reward, next_state, done):
        """Add a sample to the memory

        Args:
            state (np.array): State that needs to be added
            action (int): Action that the agent is going to take
            reward (float): Reward taking the action
            next_state (np.array): State obtained by taking action in the current state
            done (bool): Wether next_state represents an ending state
        """
        idx = self.pointer % self.buffer_size
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state

        # TODO: Why 1 - int(done)?
        self.done_mem[idx] = 1 - int(done)

        self.pointer += 1

    def sample(self, batch_size=64):
        """Get a batch of samples 

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 64.

        Returns:
            tuple: In the form (state, action, reward, next_state, done) where each element is a numpy array
        """
        # Check if there are enough element in memory according to batch size
        if self.pointer < batch_size:
            raise MemoryError("Not enough samples in memory")

        # Maximum index of the memory
        max_mem = min(self.pointer, self.buffer_size)
        # Calculate the sampling probability through proportional prioritization
        sampling_probability = (
            (self.losses[:max_mem]+self.epsilon)**self.alpha)
        # Then normalize it
        sampling_probability /= sampling_probability.sum()
        # Then feed it to np.random
        batch = np.random.choice(
            max_mem, batch_size, replace=False, p=sampling_probability)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]

        return states, actions, rewards, next_states, dones, batch
