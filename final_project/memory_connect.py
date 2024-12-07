from collections import namedtuple

import numpy as np
import torch

Batch = namedtuple(
    'Batch', ('observations', 'actions', 'rewards', 'next_observations', 'dones')
)

class ReplayMemory:
    def __init__(self, max_size, obs_size, device):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer
            - state_size: Size of the state-space features for the environment
        """
        self.max_size = max_size
        self.device = device

        # Preallocating all the required memory, for speed concerns
        # self.states = torch.empty((max_size, state_size)).to(self.device)
        self.observations = torch.empty((max_size, 1, *obs_size)).to(self.device)
        self.actions = torch.empty((max_size, 1), dtype=torch.long).to(self.device)
        self.rewards = torch.empty((max_size, 1)).to(self.device)
        # self.next_states = torch.empty((max_size, state_size)).to(self.device)
        self.next_observations = torch.empty((max_size, 1, *obs_size)).to(self.device)
        self.dones = torch.empty((max_size, 1), dtype=torch.bool).to(self.device)

        # Pointer to the current location in the circular buffer
        self.idx = 0
        # Indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add a transition to the buffer.

        :param state: 1-D np.ndarray of state-features
        :param action: Integer action
        :param reward: Float reward
        :param next_state: 1-D np.ndarray of state-features
        :param done: Boolean value indicating the end of an episode
        """
        # self.states[self.idx] = torch.tensor(state, dtype=torch.float32).to(self.device)
        self.observations[self.idx] = torch.tensor(obs, dtype=torch.float32).to(self.device)
        self.actions[self.idx] = torch.tensor(action, dtype=torch.long).to(self.device)
        self.rewards[self.idx] = torch.tensor(reward, dtype=torch.float32).to(self.device)
        # self.next_states[self.idx] = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        self.next_observations[self.idx] = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        self.dones[self.idx] = torch.tensor(done, dtype=torch.bool).to(self.device)
        
        # Circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # Update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size: Number of transitions to sample
        :rtype: Batch
        """

        # randomly sample transitions 
        batch_size = min(batch_size, self.size)
        sample_indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = Batch(#states=self.states[sample_indices],
                      observations=self.observations[sample_indices],
                      actions = self.actions[sample_indices], 
                      rewards=self.rewards[sample_indices], 
                      # next_states=self.next_states[sample_indices], 
                      next_observations=self.next_observations[sample_indices],
                      dones=self.dones[sample_indices])

        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env: Gymnasium environment
        :param num_steps: Number of steps to populate the replay memory
        """

        obs = env.reset()
        for _ in range(num_steps):
            action = np.random.choice(env.action_space)
            next_obs, reward, done = env.step(action) 
            self.add(obs, action, reward, next_obs, done)
            if done:
                obs = env.reset()
            else:
                # state = next_state
                obs = next_obs