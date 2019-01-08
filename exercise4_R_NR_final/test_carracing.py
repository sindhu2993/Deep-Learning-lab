from __future__ import print_function

import gym
from dqn.dqn_agent import DQNAgent
from train_carracing import run_episode
from dqn.networks import *
import numpy as np
import os
from datetime import datetime
import json

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped

 

    #TODO: Define networks and load agent
    # ....

    state_dim = (96,96)
    max_timesteps = 1000
    nr_actions= 7
    nr_episodes = 1500
    batch_size = 64
    history_length = 2
    skip_frames = 2 
    #load_data = False

    Q = CNN(state_dim, nr_actions, hidden=300, lr=0.0003, history_length=history_length)
    Q_target = CNNTargetNetwork(state_dim, nr_actions, hidden=300, lr=0.0003, history_length=history_length)
    agent = DQNAgent(Q, Q_target, nr_actions, discount_factor=0.99, batch_size=batch_size,epsilon=0.05,epsilon_decay=0.95, epsilon_min=0.05,tau=0.5, game='carracing',exploration="boltzmann", history_length=history_length)


    agent.load("./models_carracing/dqn_agent_1000.ckpt")
    #agent.load("/home/singhs/Downloads/exercise4_R_NR/models_carracing/dqn_agent_600.ckpt")

    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        #stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        stats, frames = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=True, do_training=False, rendering=True, skip_frames=skip_frames, history_length=history_length)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

