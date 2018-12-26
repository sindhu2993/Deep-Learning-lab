import os
from datetime import datetime
import gym
import json
from dqn.dqn_agent import DQNAgent
from train_cartpole import run_episode
from dqn.networks import *
import numpy as np

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CartPole-v0").unwrapped

    # TODO: load DQN agent
    # ...


    state_dim = 4
    num_actions = 2
    hidden = 300
    lr = 1e-3
    discount_factor = 0.95
    epsilon= 0.5
    epsilon_decay = 0.99
    epsilon_min = 0.05
    game = "cartpole"
    exploration = "greedy"
    batch_size = 32

    Q = NeuralNetwork(state_dim=state_dim, num_actions=num_actions, hidden=300, lr=1e-3)

    Q_target = TargetNetwork(state_dim=state_dim, num_actions=num_actions, hidden=300, lr=1e-3)

    agent = DQNAgent(Q, Q_target, num_actions, game=game, exploration=exploration, discount_factor=0.95, batch_size=32, epsilon=0.5, epsilon_decay=0.99, epsilon_min= 0.05)

    agent.load("models_cartpole/dqn_agent.ckpt")
 
    n_test_episodes = 15

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')

