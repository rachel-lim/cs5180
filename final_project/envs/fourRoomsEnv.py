from typing import List, Tuple
import random

import numpy as np
import matplotlib.pyplot as plt

class FourRoomsEnv():
    def __init__(self, rand_seed=None) -> None:
        if rand_seed:
            np.random.seed(rand_seed)
            random.seed(rand_seed)

        self.grid = np.array([[1., 1, 1, 1, 1, 1, 1, 1, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 1, 0, 1, 1, 1, 0, 1, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 0, 0, 0, 1],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        
        empty_cells = np.where(self.grid == 0.0)
        self.state_space = [[col, row] for row, col in zip(empty_cells[0], empty_cells[1])]
        self.num_states = len(self.state_space)

        n = self.grid.shape[0]-2
        self.start_goal_pairs = [[[1, 1], [n, n]],
                            [[1, n], [n, 1]],
                            [[n, 1], [1, n]],
                            [[n, n], [1, 1]]]
        pair = self.start_goal_pairs[np.random.choice(range(4))]
        self.start_state = pair[0]
        self.goal_state = pair[1]

        self.actions = {0: [-1, -1],  # up left
                        1: [-1, 0],  # up
                        2: [-1, 1],  # up right
                        3: [0, -1],
                        4: [0, 0],
                        5: [0, 1],
                        6: [1, -1],
                        7: [1, 0],
                        8: [1, 1]}
        
        self.action_space = list(self.actions.keys())
        self.num_actions = len(self.action_space)

        self.state = self.start_state
        self.grid[tuple(self.state)] = 0.5
        self.grid[tuple(self.goal_state)] = 0.75

    def reset(self) -> Tuple[List[int], np.array]:
        """Resets environment
        
        Returns:
            self.state: coordinates of start state in form [x, y]
            self.grid: reset maze grid
        """
        # choose new start and goal state
        pair = self.start_goal_pairs[np.random.choice(range(4))]
        self.start_state = pair[0]
        self.goal_state = pair[1]

        self.state = self.start_state  # reset state to start position
        self.grid[self.grid < 1] = 0  # reset visited cells to empty
        self.grid[tuple(self.state)] = 0.5 # current location is start state
        self.grid[tuple(self.goal_state)] = 0.75

        return self.grid

    def step(self, action: str) -> Tuple[List[int], np.array, int, bool]:
        if action not in self.action_space:
            raise ValueError(f"invalid action: {action}")
        reward = -1
        done = False
        next_state = [self.state[0] + self.actions[action][0], self.state[1] + self.actions[action][1]]

        # if next state is a wall, state doesn't change
        if self.grid[tuple(next_state)] == 1:
            next_state = self.state
        
        # reach goal
        if next_state == self.goal_state:
            reward = 10
            done = True

        self.grid[tuple(self.state)] = 0  # reset last state
        self.grid[tuple(next_state)] = 0.5  # set current state
        self.state = next_state
        
        return self.grid, reward, done

    def render(self):
        plt.imshow(self.grid)