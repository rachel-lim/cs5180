import copy
import time 

import numpy as np
import torch
import tqdm 

sys.path.append('./')
sys.path.append('..')
sys.path.append('../helping_hands_rl_envs')
from utils import parameters
from buffer import QLearningBuffer
from utils.logger import Logger
from utils.schedules import LinearSchedule
from dqn_net import CNNCom
from equivariant_dqn_net import EquivariantCNNCom
from dqn_agent_com import DQNAgentCom

from storage.per_buffer import PrioritizedQLearningBuffer, EXPERT, NORMAL
from storage.aug_buffer import QLearningBufferAug
from storage.per_aug_buffer import PrioritizedQLearningBufferAug

from utils.env_wrapper import EnvWrapper

from utils.create_agent import createAgent
import threading

from utils.torch_utils import ExpertTransition, normalizeTransition, augmentBuffer

def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_step(agent, replay_buffer, logger, p_beta_schedule):
    batch = replay_buffer.sample(parameters.batch_size)
    loss, td_error = agent.update(batch)

    logger.trainingBookkeeping(loss, td_error.mean().item())
    logger.num_training_steps += 1
    if logger.num_training_steps % parameters.target_update_freq == 0:
        agent.updateTarget()

def preTrainCURLStep(agent, replay_buffer, logger):
    batch = replay_buffer.sample(parameters.batch_size)
    loss = agent.updateCURLOnly(batch)
    logger.trainingBookkeeping(loss, 0)

def saveModelAndInfo(logger, agent):
    logger.saveModel(logger.num_steps, env, agent)
    logger.saveLearningCurve(100)
    logger.saveLossCurve(100)
    logger.saveTdErrorCurve(100)
    logger.saveExpertSampleCurve(100)
    logger.saveEvalCurve()
    logger.saveRewards()
    logger.saveLosses()
    logger.saveTdErrors()
    logger.saveEvalRewards()

def evaluate(env, agent, logger):
    state = env.reset()
    evaled = 0
    temp_reward = [[] for _ in range(parameters.num_eval_processes)]
    eval_rewards = []
    if not parameters.no_bar:
        eval_bar = tqdm(total=parameters.num_eval_episodes)
    while evaled < parameters.num_eval_episodes:
        actions_star_idx, actions_star = agent.getGreedyActions(states, obs)
        states_, obs_, rewards, dones = env.step(actions_star)
        rewards = rewards.numpy()
        dones = dones.numpy()
        states = copy.copy(states_)
        obs = copy.copy(obs_)
        for i, r in enumerate(rewards.reshape(-1)):
            temp_reward[i].append(r)
        evaled += int(np.sum(dones))
        for i, d in enumerate(dones.astype(bool)):
            if d:
                R = 0
                for r in reversed(temp_reward[i]):
                    R = r + parameters.gamma * R
                eval_rewards.append(R)
                temp_reward[i] = []
        if not parameters.no_bar:
            eval_bar.update(evaled - eval_bar.n)
    logger.eval_rewards.append(np.mean(eval_rewards[:parameters.num_eval_episodes]))
    if not parameters.no_bar:
        eval_bar.close()

def countParameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def create_agent(model):
    agent = DQNAgentCom(lr=lr, gamma=gamma, device=device, dx=dpos, dy=dpos, dz=dpos, dr=drot, n_p=n_p, n_theta=n_theta).
    if model == 'cnn':
        net = CNNCom((obs_channel, crop_size, crop_size), n_p=n_p, n_theta=n_theta).to(device)
    elif model == 'equi':
        net = EquivariantCNNCom(n_p=n_p, n_theta=n_theta, initialize=initialize).to(device)
    agent.initNetwork(net, initialize_target=not test)
    agent.aug = parameters.aug
    agent.aug_type = parameters.aug_type
    
    return agent

def train(alg, model):
    eval_thread = None
    start_time = time.time()
    set_seed(parameters.seed)

    # setup env
    print('creating envs')
    # envs = EnvWrapper(num_processes, simulator, env, env_config, planner_config)
    # eval_envs = EnvWrapper(parameters.num_eval_processes, simulator, env, env_config, planner_config)
    # TO DO

    # setup agent
    agent = createAgent()
    eval_agent = createAgent(test=True)
    # .train() is required for equivariant network
    agent.train()
    eval_agent.train()
    # if load_model_pre:
    #     agent.loadModel(load_model_pre)

    # logging
    log_dir = os.path.join(log_pre, '{}_{}'.format(alg, model))
    if note:
        log_dir += '_'
        log_dir += note

    logger = Logger(log_dir, env, 'train', num_processes, max_train_step, parameters.gamma, log_sub)
    hyper_parameters['model_shape'] = agent.getModelStr()
    logger.saveParameters(hyper_parameters)

    replay_buffer = QLearningBuffer(buffer_size)
    exploration = LinearSchedule(schedule_timesteps=explore, initial_p=init_eps, final_p=final_eps)
    p_beta_schedule = LinearSchedule(schedule_timesteps=max_train_step, initial_p=per_beta, final_p=1.0)

    if load_sub:
        logger.loadCheckPoint(os.path.join(log_dir, load_sub, 'checkpoint'), envs, agent, replay_buffer)

    if load_buffer is not None and not load_sub:
        logger.loadBuffer(replay_buffer, load_buffer, load_n)

    if planner_episode > 0 and not load_sub:
        planner_envs = envs
        planner_num_process = num_processes
        j = 0
        states, obs = planner_envs.reset()
        s = 0
        if not parameters.no_bar:
            planner_bar = tqdm(total=planner_episode)
        while j < planner_episode:
            plan_actions = planner_envs.getNextAction()
            planner_actions_star_idx, planner_actions_star = agent.getActionFromPlan(plan_actions)
            states_, obs_, rewards, dones = planner_envs.step(planner_actions_star, auto_reset=True)
            steps_lefts = planner_envs.getStepLeft()
            for i in range(planner_num_process):
                transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), planner_actions_star_idx[i].numpy(),
                                              rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
                                              steps_lefts[i].numpy(), np.array(1))
                if obs_type == 'pixel':
                    transition = normalizeTransition(transition)
                replay_buffer.add(transition)
            states = copy.copy(states_)
            obs = copy.copy(obs_)

            j += dones.sum().item()
            s += rewards.sum().item()

            if not parameters.no_bar:
                planner_bar.set_description('{:.3f}/{}, AVG: {:.3f}'.format(s, j, float(s)/j if j != 0 else 0))
                planner_bar.update(dones.sum().item())
        if expert_aug_n > 0:
            augmentBuffer(replay_buffer, buffer_aug_type, expert_aug_n)

    # pre train
    if pre_train_step > 0:
        pbar = tqdm(total=pre_train_step)
        while len(logger.losses) < pre_train_step:
            t0 = time.time()
            train_step(agent, replay_buffer, logger, p_beta_schedule)
            if logger.num_training_steps % 1000 == 0:
                logger.saveLossCurve(100)
                logger.saveTdErrorCurve(100)
            if not parameters.no_bar:
                pbar.set_description('loss: {:.3f}, time: {:.2f}'.format(float(logger.getCurrentLoss()), time.time()-t0))
                pbar.update(len(logger.losses)-pbar.n)

            if (time.time() - start_time) / 3600 > time_limit:
                logger.saveCheckPoint(args, envs, agent, replay_buffer)
                exit(0)
        pbar.close()
        logger.saveModel(0, 'pretrain', agent)

    if not parameters.no_bar:
        pbar = tqdm(total=max_train_step)
        pbar.set_description('Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0')
    timer_start = time.time()

    states, obs = envs.reset()

    # TRAINING
    while logger.num_training_steps < max_train_step:

        # get epsilon
        if fixed_eps:
            eps = final_eps
        else:
            eps = exploration.value(logger.num_training_steps)

        is_expert = 0
        actions_star_idx, actions_star = agent.getEGreedyActions(states, obs, eps)  # NEED TO CHANGE

        envs.stepAsync(actions_star, auto_reset=False)

        if len(replay_buffer) >= training_offset:
            for training_iter in range(training_iters):
                train_step(agent, replay_buffer, logger, p_beta_schedule)

        # states_, obs_, rewards, dones = envs.stepWait()
        next_state, reward, done = envs.step() # TO DO
        # steps_lefts = envs.getStepLeft()

        # done_idxes = torch.nonzero(dones).squeeze(1)
        # if done_idxes.shape[0] != 0:
        if done:
            state = envs.reset() # TO DO
            # reset_states_, reset_obs_ = envs.reset_envs(done_idxes)
            # for j, idx in enumerate(done_idxes):
            #     states_[idx] = reset_states_[j]
            #     obs_[idx] = reset_obs_[j]

        # if not alg[:2] == 'bc':  # algorithm starts with bc ??
        #     for i in range(num_processes):
        #         transition = ExpertTransition(states[i].numpy(), obs[i].numpy(), actions_star_idx[i].numpy(),
        #                                       rewards[i].numpy(), states_[i].numpy(), obs_[i].numpy(), dones[i].numpy(),
        #                                       steps_lefts[i].numpy(), np.array(is_expert))
        #         if obs_type == 'pixel':
        #             transition = normalizeTransition(transition)
        #         replay_buffer.add(transition)
        # logger.stepBookkeeping(rewards.numpy(), steps_lefts.numpy(), dones.numpy())

        states = copy.copy(states_)
        obs = copy.copy(obs_)

        if (time.time() - start_time)/3600 > time_limit:
            break

        if not parameters.no_bar:
            timer_final = time.time()
            description = 'Action Step:{}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}'.format(
                logger.num_steps, logger.getCurrentAvgReward(100), logger.eval_rewards[-1] if len(logger.eval_rewards) > 0 else 0, eps, float(logger.getCurrentLoss()),
                timer_final - timer_start)
            pbar.set_description(description)
            timer_start = timer_final
            pbar.update(logger.num_training_steps-pbar.n)
        logger.num_steps += num_processes

        if logger.num_training_steps > 0 and eval_freq > 0 and logger.num_training_steps % eval_freq == 0:
            if eval_thread is not None:
                eval_thread.join()
            eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(target=evaluate, args=(eval_envs, eval_agent, logger))
            eval_thread.start()
            # evaluate(eval_envs, agent, logger)

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    if eval_thread is not None:
        eval_thread.join()
    saveModelAndInfo(logger, agent)
    logger.saveCheckPoint(args, envs, agent, replay_buffer)
    if logger.num_training_steps >= max_train_step:
        logger.saveResult()
    # envs.close()
    # eval_envs.close()
    print('training finished')
    if not parameters.no_bar:
        pbar.close()