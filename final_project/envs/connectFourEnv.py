from typing import List, Tuple
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

class ConnectFourEnv():
    def __init__(self, rand_seed=None) -> None:
        if rand_seed:
            np.random.seed(rand_seed)
            random.seed(rand_seed)

        self.grid = np.array([[0., 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1]])

        self.action_space = list(range(self.grid.shape[1]))
        self.num_actions = len(self.action_space)

    def reset(self) -> Tuple[List[int], np.array]:
        """Resets environment
        
        Returns:
            self.state: coordinates of start state in form [x, y]
            self.grid: reset maze grid
        """
        self.grid = np.array([[0., 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1, 1, 1]])
        # self.grid = np.array([[0., 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #               [1, 1, 1, 1, 1, 1, 1, 1, 1]])

        return self.grid

    def _check_win(self, grid, diag=True):
        horizontal_kernel = np.array([[ 1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        if diag:
            detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        else:
            detection_kernels = [horizontal_kernel, vertical_kernel]

        for kernel in detection_kernels:
            if (convolve2d(grid, kernel, mode="valid") == 4).any():
                return True
        return False

    def step(self, action: str) -> Tuple[List[int], np.array, int, bool]:
        if action not in self.action_space:
            raise ValueError(f"invalid action: {action}")
        reward = -1
        done = False

        if sum(self.grid[:, action]==0) > 0:
            reward = 0
            row = np.where(self.grid[:, action]==0)[0][-1] # bottom empty row
            self.grid[row, action] = 0.5

            if self._check_win(self.grid == 0.5):
                reward = 10
                done = True
            else:
                opponent_action = -1
                # for possible_action in self.action_space:
                #     check_grid = self.grid.copy()
                #     if sum(check_grid[:, possible_action]==0) > 0: # if action is valid
                #         row = np.where(check_grid[:, possible_action]==0)[0][-1] # bottom empty row
                #         check_grid[row, possible_action] = 0.5
                #         if self._check_win(check_grid == 0.5, diag=False): # if that action would let opponent win
                #             if np.random.rand() < 0.75:
                #                 opponent_action = possible_action # 75% chance take correct move
                #             else:
                #                 opponent_action = np.random.choice(self.action_space)
                if opponent_action == -1:
                    opponent_action = np.random.choice(self.action_space)
                # if np.random.rand() < 0.75:
                #     opponent_action = action
                # else:
                #     opponent_action = np.random.choice(self.action_space)
                while sum(self.grid[:, opponent_action]==0) == 0:
                    opponent_action = np.random.choice(self.action_space)
                row = np.where(self.grid[:, opponent_action]==0)[0][-1] # bottom empty row
                self.grid[row, opponent_action] = 0.75
                if self._check_win(self.grid == 0.75):
                    reward = -10
                    done = True
                if np.sum(self.grid==0)==0:
                    done = True
        
        return self.grid, reward, done

    def render(self):
        plt.imshow(self.grid)