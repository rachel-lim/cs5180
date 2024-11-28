import collections
import copy
import os
import random
import time

import numpy as np
import torch
import tqdm

from utils import parameters
from buffer import QLearningBuffer
from utils.logger import Logger
from utils.schedules import LinearSchedule
from dqn_net import CNNCom
from equivariant_dqn_net import EquivariantCNNCom
from dqn_agent_com import DQNAgentCom
from env import MazeEnv

import threading

# from utils.torch_utils import ExpertTransition, normalizeTransition, augmentBuffer

Transition = collections.namedtuple('Transition', 'state obs action reward next_state next_obs done')


def set_seed(s):
    np.random.seed(s)
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_step(agent, replay_buffer):#, logger):
    batch = replay_buffer.sample(parameters.batch_size)
    loss, td_error = agent.update(batch)

    # logger.trainingBookkeeping(loss, td_error.mean().item())
    # logger.num_training_steps += 1
    # if logger.num_training_steps % parameters.target_update_freq == 0:
    #     agent.updateTarget()


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
    logger.eval_rewards.append(np.mean(eval_rewards[: parameters.num_eval_episodes]))
    if not parameters.no_bar:
        eval_bar.close()


def countParameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def create_agent(model, test=False):
    agent = DQNAgentCom(
        lr=parameters.lr,
        gamma=parameters.gamma,
        device=parameters.device,
        dx=parameters.dpos,
        dy=parameters.dpos,
        dz=parameters.dpos,
        dr=parameters.drot,
        n_p=parameters.n_p,
        n_theta=parameters.n_theta,
    )
    if model == "cnn":
        net = CNNCom((parameters.obs_channel, parameters.crop_size, parameters.crop_size), n_p=parameters.n_p, n_theta=parameters.n_theta).to(
            parameters.device
        )
    elif model == "equi":
        net = EquivariantCNNCom(n_p=parameters.n_p, n_theta=parameters.n_theta, initialize=parameters.initialize).to(
            parameters.device
        )
    agent.initNetwork(net, initialize_target=not test)
    agent.aug = parameters.aug
    agent.aug_type = parameters.aug_type

    return agent


def train(model):
    eval_thread = None
    start_time = time.time()
    set_seed(parameters.seed)

    # setup env
    print("creating envs")
    env = MazeEnv(dim=10)
    
    # setup agent
    agent = create_agent(model)
    eval_agent = create_agent(model, test=True)
    # .train() is required for equivariant network
    if model == "equi":
        agent.train()
        eval_agent.train()

    # logging
    # log_dir = os.path.join(log_pre, "{}_{}".format(alg, model))

    # logger = Logger(
    #     log_dir, env, "train", parameters.num_processes, parameters.max_train_step, parameters.gamma, parameters.log_sub
    # )
    # hyper_parameters['model_shape'] = agent.getModelStr()
    # logger.saveParameters(hyper_parameters)

    replay_buffer = QLearningBuffer(parameters.buffer_size)
    exploration = LinearSchedule(
        schedule_timesteps=parameters.explore, initial_p=parameters.init_eps, final_p=parameters.final_eps
    )

    # load checkpoint (optional)
    # if parameters.load_sub:
    #     logger.loadCheckPoint(
    #         os.path.join(log_dir, parameters.load_sub, "checkpoint"), env, agent, replay_buffer
    #     )

    # load buffer (optional)
    # if parameters.load_buffer is not None and not parameters.load_sub:  # default None
    #     logger.loadBuffer(replay_buffer, parameters.load_buffer, load_n)

    # prepopulate replay buffer
    if parameters.planner_episode > 0 and not parameters.load_sub:
        planner_env = env
        planner_num_process = parameters.num_processes  # ?? default 5
        j = 0
        state, obs = planner_env.reset()  # TO DO
        s = 0
        if not parameters.no_bar:
            planner_bar = tqdm(total=parameters.planner_episode)
        while j < parameters.planner_episode:
            action = planner_env.get_egreedy_action() # TO DO
            next_state, next_obs, reward, done = planner_env.step(action)  # TO DO
            transition = Transition(state, obs, action, reward, next_state, next_obs, done)  # left out steps_left, np.array(1)
            replay_buffer.add(transition)

            state = copy.copy(next_state)
            obs = copy.copy(next_obs)

            j += done
            s += reward

            if not parameters.no_bar:
                planner_bar.set_description(
                    "{:.3f}/{}, AVG: {:.3f}".format(s, j, float(s) / j if j != 0 else 0)
                )
                planner_bar.update(dones.sum().item())

    # removed pre-train

    # if not parameters.no_bar:
    #     pbar = tqdm(total=parameters.max_train_step)
    #     pbar.set_description("Episodes:0; Reward:0.0; Explore:0.0; Loss:0.0; Time:0.0")
    # timer_start = time.time()

    state, obs = env.reset()

    # TRAINING
    for t in range(parameters.max_train_step):

        # get epsilon
        if parameters.fixed_eps:
            eps = parameters.final_eps
        else:
            eps = exploration.value(logger.num_training_steps)

        # get egreedy actions
        if np.random.rand() < eps:
            action = np.random.choice(env.action_space)
        else:
            action = agent.get_greedy_action(state, obs, eps) # TO DO

        # envs.stepAsync(actions_star, auto_reset=False)  # ??
        next_state, next_obs, reward, done = env.step()

        # train on batch
        if len(replay_buffer) >= parameters.training_offset:
            train_step(agent, replay_buffer)#, logger)
        
        # update target
        if t % parameters.target_update_freq == 0:
            agent.updateTarget()

        if done:
            state, obs = env.reset()

        state = copy.copy(next_state)
        obs = copy.copy(next_obs)

        # if not parameters.no_bar:
        #     timer_final = time.time()
        #     description = "Action Step:{}; Reward:{:.03f}; Eval Reward:{:.03f}; Explore:{:.02f}; Loss:{:.03f}; Time:{:.03f}".format(
        #         logger.num_steps,
        #         logger.getCurrentAvgReward(100),
        #         logger.eval_rewards[-1] if len(logger.eval_rewards) > 0 else 0,
        #         eps,
        #         float(logger.getCurrentLoss()),
        #         timer_final - timer_start,
        #     )
        #     pbar.set_description(description)
        #     timer_start = timer_final
        #     pbar.update(logger.num_training_steps - pbar.n)
        # logger.num_steps += num_processes

        # ??
        if (
            logger.num_training_steps > 0
            and eval_freq > 0
            and logger.num_training_steps % eval_freq == 0
        ):
            if eval_thread is not None:
                eval_thread.join()
            eval_agent.copyNetworksFrom(agent)
            eval_thread = threading.Thread(
                target=evaluate, args=(eval_envs, eval_agent, logger)
            )
            eval_thread.start()
            # evaluate(eval_envs, agent, logger)

        if logger.num_steps % (num_processes * save_freq) == 0:
            saveModelAndInfo(logger, agent)

    if eval_thread is not None:
        eval_thread.join()
    # saveModelAndInfo(logger, agent)
    # logger.saveCheckPoint(args, env, agent, replay_buffer)
    
    # if logger.num_training_steps >= parameters.max_train_step:
    #     logger.saveResult()
    print("training finished")
    if not parameters.no_bar:
        pbar.close()
