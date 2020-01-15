
from sc2 import Race, Difficulty
from sc2.player import Bot, Computer
import sc2
import sys
import argparse
import os
sys.path.insert(0, os.path.abspath('../'))
from runfiles.sc_runner import StarmniBot
from runfiles.minigame_runner import SC2MicroBot
from agents.prolonet_agent import DeepProLoNet
from agents.non_deep_prolonet_agent import ShallowProLoNet
from agents.random_prolonet_agent import RandomProLoNet
from agents.lstm_agent import LSTMNet
from agents.baseline_agent import FCNet


def run_episode(main_agent, self_play=False):
    agent_in = main_agent.duplicate()

    bot = StarmniBot(rl_agent=agent_in)
    bot2 = StarmniBot(rl_agent=main_agent)
    opponents = [
        Computer(Race.Zerg, Difficulty.Easy),
        Bot(Race.Protoss, bot2)
    ]
    if self_play:
        enemy = opponents[1]
    else:
        enemy = opponents[0]

    try:
        result = sc2.run_game(sc2.maps.get("Acid Plant LE"),
                              [Bot(Race.Protoss, bot), enemy],
                              realtime=False)
    except KeyboardInterrupt:
        result = [-1, -1]
    except Exception as e:
        print(str(e))
        print("No worries", e, " carry on please")
    return 0


def run_mini_episode(main_agent, self_play=False):
    print(main_agent.action_network.state_dict())
    bot = SC2MicroBot(rl_agent=main_agent)

    try:
        result = sc2.run_game(sc2.maps.get("FindAndDefeatZerglings"),
                              [Bot(Race.Protoss, bot)],
                              realtime=True)
    except KeyboardInterrupt:
        result = [-1, -1]
    except Exception as e:
        print(str(e))
        print("No worries", e, " carry on please")
    if type(result) == list and len(result) > 1:
        result = result[0]
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='fc')
    parser.add_argument("-adv", "--adversary",
                        help="for prolonet, init as adversarial? true for yes, false for no",
                        type=bool, default=False)
    parser.add_argument("-s", "--sl_init", help="sl to rl for fc net?", type=bool, default=False)
    parser.add_argument('--self', dest='self_play', action='store_true', default=False)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='macro')

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'shallow_prolo', 'prolo', 'random', 'fc', 'lstm'
    ADVERSARIAL = args.adversary  # Adversarial prolo, applies for AGENT_TYPE=='shallow_prolo'
    SL_INIT = args.sl_init  # SL->RL fc, applies only for AGENT_TYPE=='fc'
    self_play = args.self_play
    ENV_TYPE = args.env_type

    if ENV_TYPE == 'macro':
        dim_in = 194
        dim_out = 44
        bot_name = AGENT_TYPE + 'SC_Macro'
    elif ENV_TYPE == 'micro':
        dim_in = 32
        dim_out = 10
        bot_name = AGENT_TYPE + 'SC_Micro'

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
    policy_agent.load('../models/FINAL')
    if ENV_TYPE == 'macro':
        run_episode(policy_agent, self_play=self_play)
    elif ENV_TYPE == 'micro':
        run_mini_episode(policy_agent, self_play=self_play)
