# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
from utils import EpisodeStats
from utils import *


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=True, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)

        action_id = agent.act(state=state, deterministic=deterministic)
        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        early_done, punishment = agent.check_early_stop (reward, stats.episode_reward)
        if early_done:
            reward += punishment

        terminal = terminal or early_done
        stats.episode_reward +=reward
       

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats, step


def train_online(env, agent, num_episodes, history_length=0, skip_frames=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward","det_episode_reward", "straight", "left", "right", "accel", "brake"])

    eval_reward = []

    for i in range(num_episodes):
        #print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
       
        stats, frames = run_episode(env, agent, max_timesteps=1000, deterministic=False, do_training=True, rendering=True, skip_frames=skip_frames, history_length=history_length)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(STRAIGHT),
                                                      "left" : stats.get_action_usage(LEFT),
                                                      "right" : stats.get_action_usage(RIGHT),
                                                      "accel" : stats.get_action_usage(ACCELERATE),
                                                      "brake" : stats.get_action_usage(BRAKE)
                                                      })

        # TODO: evaluate agent with deterministic actions from time to time
        # ...

        

        if i % 100 == 0 or (i >= num_episodes - 1):
            e_reward = 0
            for i in range(5):
                eval, frame = run_episode(env, agent, deterministic=True, do_training=False, history_length=history_length)
                e_reward += eval.episode_reward
            eval_reward.append(e_reward/5)
            tensorboard.write_det_episode_data(i, eval_dict={  "det_episode_reward" : det_rewards / 5 } )

            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent_1000.ckpt")) 

        print('episode %d' %i, 'reward: ', stats.episode_reward, 'time steps : ', frames)   

        
        for i in eval_reward:
            if i >= num_episodes - 1:
                agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent_1000.ckpt"))
                break


    path = os.path.join("./", "reward_eval_training")
    os.makedirs(path)

    file_name = os.path.join(path, "reward_eval_training_%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S"))

    file_json = open(file_name, "w")
    json.dump(eval_reward, file_json)
    file_json.close()

    print('evaluation reward: ', eval_reward)
    tensorboard.close_session()


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped
    
    # TODO: Define Q network, target network and DQN agent
    # ...
    
    #state_dim = env.observation_space.shape
    state_dim = (96,96)
    nr_actions= 7
    nr_episodes = 1500
    batch_size = 64
    history_length = 2
    skip_frames = 2 
    

    Q = CNN(state_dim, nr_actions, hidden=300, lr=0.0003, history_length=history_length)
    Q_target = CNNTargetNetwork(state_dim, nr_actions, hidden=300, lr=0.0003, history_length=history_length)
    agent = DQNAgent(Q, Q_target, nr_actions, discount_factor=0.95, batch_size=batch_size,epsilon=0.05,epsilon_decay=0.95, epsilon_min=0.05,tau=0.5, game='carracing',exploration="boltzmann", history_length=history_length)

    train_online(env, agent, nr_episodes, history_length=history_length, skip_frames=skip_frames, model_dir="./models_carracing")



