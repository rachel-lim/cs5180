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
    def __init__(self, dim: int, rand_seed=None, grid=None, rotate=0) -> None:
        if rand_seed:
            np.random.seed(rand_seed)
            random.seed(rand_seed)

        if grid is None:
            self.grid = create_maze(dim)
            self.start_state = [1, 0]
            self.goal_state = [2*dim-1, 2*dim]
        else:
            self.grid = np.rot90(np.floor(np.copy(grid)), rotate)  # floor to set start cell to 0
            tmp_grid = np.zeros(self.grid.shape)
            tmp_grid[1, 0] = 1
            tmp_grid[len(grid)-2, len(grid)-1] = 2
            self.start_state = list(np.argwhere(np.rot90(tmp_grid, rotate)==1)[0])
            self.goal_state = list(np.argwhere(np.rot90(tmp_grid, rotate)==2)[0])
            
        empty_cells = np.where(self.grid == 0.0)
        self.state_space = [[col, row] for row, col in zip(empty_cells[0], empty_cells[1])]
        self.num_states = len(self.state_space)

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

        self.t = 0 

    def reset(self) -> Tuple[List[int], np.array]:
        """Resets environment
        
        Returns:
            self.state: coordinates of start state in form [x, y]
            self.grid: reset maze grid
        """
        self.state = self.start_state  # reset state to start position
        self.grid[self.grid == 0.5] = 0  # reset visited cells to empty
        self.grid[tuple(self.state)] = 0.5 # current location is start state
        self.t = 0

        return self.grid

    def step(self, action: str) -> Tuple[List[int], np.array, int, bool]:
        if action not in self.action_space:
            raise ValueError(f"invalid action: {action}")
        reward = -1
        done = False
        next_state = [self.state[0] + self.actions[action][0], self.state[1] + self.actions[action][1]]

        # if leaving maze, state doesn't change
        if not 0 <= next_state[0] <= len(self.grid)-1:
            next_state = self.state
        if not 0 <= next_state[1] <= len(self.grid)-1:
            next_state = self.state

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
        self.t += 1
        
        return self.grid, reward, done

    def render(self):
        plt.imshow(self.grid)