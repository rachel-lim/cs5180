import numpy as np
import torch
import torch.nn.functional as F
from utils import torch_utils

class BaseAgent:
    """
    The base RL agent class
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32):
        self.lr = lr
        self.gamma = gamma
        self.device = device
        # magnitude of actions
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dr = dr

        self.networks = []
        self.target_networks = []
        self.optimizers = []

        self.loss_calc_dict = {}

        self.aug = False
        self.aug_type = 'se2'

    def update(self, batch):
        """
        Perform a training step
        :param batch: the sampled minibatch
        :return: loss
        """
        raise NotImplementedError

    def getEGreedyActions(self, state, obs, eps):
        """
        Get e-greedy actions
        :param state: gripper holding state
        :param obs: observation
        :param eps: epsilon
        :return: action
        """
        raise NotImplementedError

    def getGreedyActions(self, state, obs):
        """
        Get greedy actions
        :param state: gripper holding state
        :param obs: observation
        :return: action
        """
        return self.getEGreedyActions(state, obs, 0)

    def _loadBatchToDevice(self, batch):
        """
        Load batch into pytorch tensor
        :param batch: list of transitions
        :return: states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, next_obs_tensor,
                 non_final_masks, step_lefts_tensor, is_experts_tensor
        """
        if self.aug:
            # perform augmentation for RAD
            batch = list(map(augmentTransition, batch, repeat(self.aug_type)))

        states = []
        images = []
        xys = []
        rewards = []
        next_states = []
        next_obs = []
        dones = []
        step_lefts = []
        is_experts = []
        for d in batch:
            states.append(d.state)
            images.append(d.obs)
            xys.append(d.action)
            rewards.append(d.reward.squeeze())
            next_states.append(d.next_state)
            next_obs.append(d.next_obs)
            dones.append(d.done)
            step_lefts.append(d.step_left)
            is_experts.append(d.expert)
        states_tensor = torch.tensor(np.stack(states)).long().to(self.device)
        obs_tensor = torch.tensor(np.stack(images)).to(self.device)
        if len(obs_tensor.shape) == 3:
            obs_tensor = obs_tensor.unsqueeze(1)
        action_tensor = torch.tensor(np.stack(xys)).to(self.device)
        rewards_tensor = torch.tensor(np.stack(rewards)).to(self.device)
        next_states_tensor = torch.tensor(np.stack(next_states)).long().to(self.device)
        next_obs_tensor = torch.tensor(np.stack(next_obs)).to(self.device)
        if len(next_obs_tensor.shape) == 3:
            next_obs_tensor = next_obs_tensor.unsqueeze(1)
        dones_tensor = torch.tensor(np.stack(dones)).int()
        non_final_masks = (dones_tensor ^ 1).float().to(self.device)
        step_lefts_tensor = torch.tensor(np.stack(step_lefts)).to(self.device)
        is_experts_tensor = torch.tensor(np.stack(is_experts)).bool().to(self.device)

        if obs_type is 'pixel':
            # scale observation from int to float
            obs_tensor = obs_tensor/255*0.4
            next_obs_tensor = next_obs_tensor/255*0.4

        self.loss_calc_dict['batch_size'] = len(batch)
        self.loss_calc_dict['states'] = states_tensor
        self.loss_calc_dict['obs'] = obs_tensor
        self.loss_calc_dict['action_idx'] = action_tensor
        self.loss_calc_dict['rewards'] = rewards_tensor
        self.loss_calc_dict['next_states'] = next_states_tensor
        self.loss_calc_dict['next_obs'] = next_obs_tensor
        self.loss_calc_dict['non_final_masks'] = non_final_masks
        self.loss_calc_dict['step_lefts'] = step_lefts_tensor
        self.loss_calc_dict['is_experts'] = is_experts_tensor

        return states_tensor, obs_tensor, action_tensor, rewards_tensor, next_states_tensor, \
               next_obs_tensor, non_final_masks, step_lefts_tensor, is_experts_tensor

    def _loadLossCalcDict(self):
        """
        Get the loaded batch data in self.loss_calc_dict
        :return: batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts
        """
        batch_size = self.loss_calc_dict['batch_size']
        states = self.loss_calc_dict['states']
        obs = self.loss_calc_dict['obs']
        action_idx = self.loss_calc_dict['action_idx']
        rewards = self.loss_calc_dict['rewards']
        next_states = self.loss_calc_dict['next_states']
        next_obs = self.loss_calc_dict['next_obs']
        non_final_masks = self.loss_calc_dict['non_final_masks']
        step_lefts = self.loss_calc_dict['step_lefts']
        is_experts = self.loss_calc_dict['is_experts']
        return batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts

    def train(self):
        """
        Call .train() for all models
        """
        for i in range(len(self.networks)):
            self.networks[i].train()
        for i in range(len(self.target_networks)):
            self.target_networks[i].train()

    def eval(self):
        """
        Call .eval() for all models
        """
        for i in range(len(self.networks)):
            self.networks[i].eval()

    def getModelStr(self):
        """
        Get the str of all models (for logging)
        :return: str of all models
        """
        return str(self.networks)

    def updateTarget(self):
        """
        Hard update the target networks
        """
        for i in range(len(self.networks)):
            self.target_networks[i].load_state_dict(self.networks[i].state_dict())

    def loadModel(self, path_pre):
        """
        Load the saved models
        :param path_pre: path prefix to the model
        """
        for i in range(len(self.networks)):
            path = path_pre + '_{}.pt'.format(i)
            print('loading {}'.format(path))
            self.networks[i].load_state_dict(torch.load(path))
        self.updateTarget()

    def saveModel(self, path_pre):
        """
        Save the models with path prefix path_pre. a '_q{}.pt' suffix will be added to each model
        :param path_pre: path prefix
        """
        for i in range(len(self.networks)):
            torch.save(self.networks[i].state_dict(), '{}_{}.pt'.format(path_pre, i))

    def getSaveState(self):
        """
        Get the save state for checkpointing. Include network states, target network states, and optimizer states
        :return: the saving state dictionary
        """
        state = {}
        for i in range(len(self.networks)):
            self.networks[i].to('cpu')
            state['{}'.format(i)] = self.networks[i].state_dict()
            state['{}_optimizer'.format(i)] = self.optimizers[i].state_dict()
            self.networks[i].to(self.device)
        for i in range(len(self.target_networks)):
            self.target_networks[i].to('cpu')
            state['{}_target'.format(i)] = self.target_networks[i].state_dict()
            self.target_networks[i].to(self.device)
        return state

    def loadFromState(self, save_state):
        """
        Load from a save_state
        :param save_state: the loading state dictionary
        """
        for i in range(len(self.networks)):
            self.networks[i].to('cpu')
            self.networks[i].load_state_dict(save_state['{}'.format(i)])
            self.networks[i].to(self.device)
            self.optimizers[i].load_state_dict(save_state['{}_optimizer'.format(i)])
        for i in range(len(self.target_networks)):
            self.target_networks[i].to('cpu')
            self.target_networks[i].load_state_dict(save_state['{}_target'.format(i)])
            self.target_networks[i].to(self.device)

    def copyNetworksFrom(self, from_agent):
        """
        Copy networks from another agent
        :param from_agent: the agent being copied from
        """
        for i in range(len(self.networks)):
            self.networks[i].load_state_dict(from_agent.networks[i].state_dict())
            
class DQNBase(BaseAgent):
    """
    Base class for DQN agent
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/16, n_p=1, n_theta=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr)

        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        # discrete action spaces
        self.p_range = torch.tensor([1])
        if n_p == 2:
            self.p_range = torch.tensor([0, 1])

        self.d_theta_range = torch.tensor([0])
        if n_theta == 3:
            self.d_theta_range = torch.tensor([-dr, 0, dr])

        self.dxy_range = torch.tensor([[-dx, -dy], [-dx, 0], [-dx, dy],
                                       [0, -dy], [0, 0], [0, dy],
                                       [dx, -dy], [dx, 0], [dx, dy]])
        self.dz_range = torch.tensor([-dz, 0, dz])

    def targetSoftUpdate(self):
        """Soft-update: target = tau*local + (1-tau)*target."""
        tau = 1e-2

        for t_param, l_param in zip(
                self.target_net.parameters(), self.policy_net.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def updateTarget(self):
        """
        Disable the default hard target update
        """
        pass

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        """
        Forward pass the Q-network
        :param state: gripper state
        :param obs: observation
        :param target_net: whether to use the target network
        :param to_cpu: move output to cpu
        :return: the output of the Q-network
        """
        raise NotImplementedError

    def calcTDLoss(self):
        """
        Calculate the TD loss
        :return: TD loss
        """
        raise NotImplementedError

    def initNetwork(self, network, initialize_target=True):
        """
        Initialize networks
        :param network: Q-network
        :param initialize_target: whether to initialize the target network
        """
        self.policy_net = network
        if initialize_target:
            self.target_net = deepcopy(network)
            self.target_networks.append(self.target_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.networks.append(self.policy_net)
        self.optimizers.append(self.optimizer)

    def decodeActions(self, p_id, dxy_id, dz_id, dtheta_id):
        """
        Get actions from the action ids
        :param p_id: gripper primitive id
        :param dxy_id: delta xy id
        :param dz_id: delta z id
        :param dtheta_id: delta theta id
        :return: action ids, actions
        """
        p = self.p_range[p_id]
        dxy = self.dxy_range[dxy_id]
        dz = self.dz_range[dz_id]
        dtheta = self.d_theta_range[dtheta_id]
        actions = torch.stack([p, dxy[:, 0], dxy[:, 1], dz, dtheta], dim=1)
        action_idxes = torch.stack([p_id, dxy_id, dz_id, dtheta_id], dim=1)
        return action_idxes, actions

    def getActionFromPlan(self, plan):
        """
        Get action ids and actions from planner actions
        :param plan: planner actions
        :return: action ids, actions
        """
        primitive = plan[:, 0:1]
        dxy = plan[:, 1:3]
        dz = plan[:, 3:4]
        dr = plan[:, 4:5]

        p_id = torch.argmin(torch.abs(self.p_range - primitive), 1)
        dxy_id = torch.argmin((dxy.unsqueeze(1) - self.dxy_range).abs().sum(2), 1)
        dz_id = torch.argmin(torch.abs(self.dz_range - dz), 1)
        dtheta_id = torch.argmin(torch.abs(self.d_theta_range - dr), 1)

        return self.decodeActions(p_id, dxy_id, dz_id, dtheta_id)

    def update(self, batch):
        """
        Perform a training step
        :param batch: the sampled minibatch
        :return: td loss, td error
        """
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        self.targetSoftUpdate()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error

class DQNAgentCom(DQNBase):
    """
    Class for DQN (composed) agent
    """
    def __init__(self, lr=1e-4, gamma=0.95, device='cuda', dx=0.005, dy=0.005, dz=0.005, dr=np.pi/32, n_p=1, n_theta=1):
        super().__init__(lr, gamma, device, dx, dy, dz, dr, n_p, n_theta)
        self.n_xy = 9
        self.n_z = 3
        self.n_theta = n_theta
        self.n_p = n_p

    def forwardNetwork(self, state, obs, target_net=False, to_cpu=False):
        """
        Forward pass the Q-network
        :param state: gripper state
        :param obs: observation
        :param target_net: whether to use the target network
        :param to_cpu: move output to cpu
        :return: the output of the Q-network
        """
        if target_net:
            net = self.target_net
        else:
            net = self.policy_net

        state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(1, 1, obs.shape[2], obs.shape[3])
        stacked = torch.cat([obs, state_tile], dim=1)
        q = net(stacked.to(self.device))
        if to_cpu:
            q = q.to('cpu')
        q = q.reshape(state.shape[0], self.n_xy, self.n_z, self.n_theta, self.n_p)
        return q

    def getEGreedyActions(self, state, obs, eps):
        """
        Get e-greedy actions
        :param state: gripper holding state
        :param obs: observation
        :param eps: epsilon
        :return: action ids, actions
        """
        with torch.no_grad():
            q = self.forwardNetwork(state, obs, to_cpu=True)
            argmax = torch_utils.argmax4d(q)
            dxy_id = argmax[:, 0]
            dz_id = argmax[:, 1]
            dtheta_id = argmax[:, 2]
            p_id = argmax[:, 3]

        rand = torch.tensor(np.random.uniform(0, 1, obs.size(0)))
        rand_mask = rand < eps
        rand_p = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_p)
        p_id[rand_mask] = rand_p.long()
        rand_dxy = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_xy)
        dxy_id[rand_mask] = rand_dxy.long()
        rand_dz = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_z)
        dz_id[rand_mask] = rand_dz.long()
        rand_dtheta = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.n_theta)
        dtheta_id[rand_mask] = rand_dtheta.long()
        return self.decodeActions(p_id, dxy_id, dz_id, dtheta_id)

    def calcTDLoss(self):
        """
        Calculate the TD loss
        :return: td loss, td error
        """
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        p_id = action_idx[:, 0]
        dxy_id = action_idx[:, 1]
        dz_id = action_idx[:, 2]
        dtheta_id = action_idx[:, 3]

        with torch.no_grad():
            q_all_prime = self.forwardNetwork(next_states, next_obs, target_net=True)
            q_prime = q_all_prime.reshape(batch_size, -1).max(1)[0]
            q_target = rewards + self.gamma * q_prime * non_final_masks

        q = self.forwardNetwork(states, obs)
        q_pred = q[torch.arange(batch_size), dxy_id, dz_id, dtheta_id, p_id]
        self.loss_calc_dict['q_output'] = q
        self.loss_calc_dict['q_pred'] = q_pred
        td_loss = F.smooth_l1_loss(q_pred, q_target)
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)
        return td_loss, td_error