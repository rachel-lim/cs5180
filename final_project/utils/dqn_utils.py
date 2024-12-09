import numpy as np
import torch
import torch.nn.functional as F

def evaluate(dqn_model, env, device, eps=0.1, timeout=500, gamma=0.99):
    """Run an evaluation episode.

    Args:
        dqn_model: the model to evaluate
        env: the environment to evaluate it in
        device: torch device
        eps: epsilon value
        timeout: max number of time steps to run an episode for
        gamma: discount factor
    """
    obs = env.reset()
    done = False
    rewards = []
    actions = []
    
    for _ in range(timeout):
        # get epsilon greedy action
        if np.random.rand() < eps:
            action = np.random.choice(env.action_space)
        else:
            obs_tensor = torch.tensor(obs.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            q_values = dqn_model(obs_tensor)  # add dim to observation
            max_q_idx = torch.where(q_values == q_values.max())[0]
            action = np.random.choice(max_q_idx.tolist())
    
        obs, reward, done = env.step(action)
        rewards.append(reward)

        if done:
            break
    g = 0
    for r in rewards[::-1]:
        g = 0.99*g + r
    return g

def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
    """Perform a single batch-update step on the given DQN model.

    Args:
        optimizer: nn.optim.Optimizer instance
        batch: Batch of experiences (class defined earlier)
        dqn_model: The DQN model to be trained
        dqn_target: The target DQN model, ~NOT~ to be trained
        gamma: The discount factor
    """
  
    values = dqn_model(batch.observations).gather(1, batch.actions)

    with torch.no_grad():
        next_q = dqn_target(batch.next_observations).max(1)[0].view(-1, 1)
        target_values = batch.rewards + gamma * next_q * (~batch.dones)

    loss = F.smooth_l1_loss(values, target_values)  # CHANGED

    optimizer.zero_grad()  # Reset all previous gradients
    loss.backward()  # Compute new gradients
    optimizer.step()  # Perform one gradient-descent step

    return loss.item()