from typing import List, Tuple
import random

import numpy as np
import matplotlib.pyplot as plt

def create_maze(dim: int) -> np.array:
    """Create maze of side length dim (actual side length 2*dim+1 because of walls) 
    From https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96
    
    Args:
        dim: side length
    
    Returns:
        np array of maze where 0 are free spaces and 1 are walls
    """
    # Create a grid filled with walls
    maze = np.ones((dim*2+1, dim*2+1))

    # Define the starting point
    x, y = (0, 0)
    maze[2*x+1, 2*y+1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if nx >= 0 and ny >= 0 and nx < dim and ny < dim and maze[2*nx+1, 2*ny+1] == 1:
                maze[2*nx+1, 2*ny+1] = 0
                maze[2*x+1+dx, 2*y+1+dy] = 0
                stack.append((nx, ny))
                break
        else:
            stack.pop()
            
    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0

    return maze

class MazeEnv():
    def __init__(self, dim: int) -> None:
        self.grid = create_maze(dim)
        self.start_state = [1, 0]
        self.goal_state = [19, 20]

        empty_cells = np.where(self.grid == 0.0)
        self.state_space = [[col, row] for row, col in zip(empty_cells[0], empty_cells[1])]
        self.num_states = len(self.state_space)

        self.actions = {"up": [-1, 0],
                        "down": [1, 0],
                        "left": [0, -1],
                        "right": [0, 1]}

        self.action_space = list(self.actions.keys())
        self.num_actions = len(self.action_space)

        self.state = self.start_state
        self.grid[tuple(self.state)] = 0.5

    def reset(self) -> Tuple[List[int], np.array]:
        """Resets environment
        
        Returns:
            self.state: coordinates of start state in form [x, y]
            self.grid: reset maze grid
        """
        self.state = self.start_state  # reset state to start position
        self.grid[self.grid == 0.5] = 0  # reset visited cells to empty
        self.grid[tuple(self.state)] = 0.5 # visited start state

        return self.state, self.grid

    def step(self, action: str) -> Tuple[List[int], np.array, int, bool]:
        if action not in self.action_space:
            raise ValueError(f"invalid action: {action}")
        reward = 0
        done = False
        next_state = [self.state[0] + self.actions[action][0], self.state[1] + self.actions[action][1]]

        # if next state is a wall, state doesn't change
        if self.grid[tuple(next_state)] == 0:
            next_state = self.state
            reward = -1
        
        # if leaving maze (left from start), state doesn't change
        if next_state[1] == -1:
            next_state = self.state
            reward = -1
        
        # reach goal
        if next_state == self.goal_state:
            reward = 1
            done = True

        self.grid[tuple(next_state)] = 0.5
        self.state = next_state
        
        return next_state, self.grid, reward, done

    def render(self):
        plt.imshow(self.grid)