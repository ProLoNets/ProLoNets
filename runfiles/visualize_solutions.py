# Created by Andrew Silva on 12/27/18
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
import argparse
import os
sys.path.insert(0, os.path.abspath('../'))
from agents.prolonet_agent import DeepProLoNet
from agents.non_deep_prolonet_agent import ShallowProLoNet
from agents.random_prolonet_agent import RandomProLoNet
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet
import random
import time


AGENT_TYPE = 'tree'
ENV_TYPE = 'lunar'
EP_NUM = 500


def run_episode(q, agent_in):
    agent = agent_in.duplicate()
    if ENV_TYPE == 'lunar':
        env = gym.make('LunarLander-v2')
    else:
        env = gym.make('CartPole-v1')
    env = gym.wrappers.Monitor(env, './', force=True)
    state = env.reset()  # Reset environment and record the starting state
    done = False

    for _ in range(1000):
        action = agent.get_action(state)
        # Step through environment using chosen action
        state, reward, done, __ = env.step(action)
        # Save reward
        print(state)
        agent.save_reward(reward)
        if done:
            env.close()
            break
        env.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='fc')
    parser.add_argument("-adv", "--adversary",
                        help="for prolonet, init as adversarial? true for yes, false for no",
                        type=bool, default=False)
    parser.add_argument("-s", "--sl_init", help="sl to rl for fc net?", type=bool, default=False)
    parser.add_argument("-dm", "--deepen_method", help="how to deepen?", type=str, default='random')
    parser.add_argument("-dc", "--deepen_criteria", help="when to deepen?", type=str, default='entropy')
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='cart')

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'shallow_prolo', 'prolo', 'random', 'fc', 'lstm'
    ADVERSARIAL = args.adversary  # Adversarial prolo, applies for AGENT_TYPE=='shallow_prolo'
    SL_INIT = args.sl_init  # SL->RL fc, applies only for AGENT_TYPE=='fc'
    DEEPEN_METHOD = args.deepen_method  # 'random', 'fc', 'parent', method for deepening, only applies for AGENT_TYPE=='prolo'
    DEEPEN_CRITERIA = args.deepen_criteria  # 'entropy', 'num', 'value', criteria for when to deepen. AGENT_TYPE=='prolo' only
    ENV_TYPE = args.env_type
    if ENV_TYPE == 'lunar':
        init_env = gym.make('LunarLander-v2')
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
    else:
        init_env = gym.make('CartPole-v1')
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
    init_env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    bot_name = AGENT_TYPE + ENV_TYPE

    if AGENT_TYPE == 'prolo':
        policy_agent = DeepProLoNet(distribution='one_hot',
                                    bot_name=bot_name,
                                    input_dim=dim_in,
                                    output_dim=dim_out,
                                    deepen_method='random',
                                    deepen_criteria='entropy')
    elif AGENT_TYPE == 'fc':
        policy_agent = FCNet(input_dim=dim_in,
                             bot_name=bot_name,
                             output_dim=dim_out,
                             sl_init=SL_INIT)
    elif AGENT_TYPE == 'random':
        policy_agent = RandomProLoNet(input_dim=dim_in,
                                      bot_name=bot_name,
                                      output_dim=dim_out)
    elif AGENT_TYPE == 'lstm':
        policy_agent = LSTMNet(input_dim=dim_in,
                               bot_name=bot_name,
                               output_dim=dim_out)
    elif AGENT_TYPE == 'shallow_prolo':
        policy_agent = ShallowProLoNet(distribution='one_hot',
                                       input_dim=dim_in,
                                       bot_name=bot_name,
                                       output_dim=dim_out,
                                       adversarial=ADVERSARIAL)
    else:
        raise Exception('No valid network selected')
    fn = '../models/'+str(EP_NUM)+'th'+AGENT_TYPE+ENV_TYPE+'.pth.tar'
    policy_agent.load('../models/FINAL')

    for _ in range(1):
        run_episode(None, policy_agent)
