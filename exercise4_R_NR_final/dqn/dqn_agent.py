import tensorflow as tf
import numpy as np
import random 
from dqn.replay_buffer import ReplayBuffer

class DQNAgent:

    def __init__(self, Q, Q_target, num_actions, discount_factor=0.99, batch_size=64, epsilon=0.05,epsilon_decay=1, epsilon_min=0.05,tau=1, game='cartpole',exploration="epsilon_greedy", history_length=0) :#, load_data=False):
        """
         Q-Learning agent for off-policy TD control using Function Approximation.
         Finds the optimal greedy policy while following an epsilon-greedy policy.

         Args:
            Q: Action-Value function estimator (Neural Network)
            Q_target: Slowly updated target network to calculate the targets.
            num_actions: Number of actions of the environment.
            discount_factor: gamma, discount factor of future rewards.
            batch_size: Number of samples per batch.
            epsilon: Chance to sample a random action. Float betwen 0 and 1.
        """
        self.Q = Q      
        self.Q_target = Q_target
        self.game = game

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.epsilon_min = epsilon_min

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        self.exploration = exploration

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # Start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        # TODO:
        # 1. add current transition to replay buffer
        # 2. sample next batch and perform batch update: 
        #       2.1 compute td targets: 
        #              td_target =  reward + discount * max_a Q_target(next_state_batch, a)
        #       2.2 update the Q network
        #              self.Q.update(...)
        #       2.3 call soft update for target network
        #              self.Q_target.update(...)
   
        '''
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)        
        states, actions, next_states, rewards, dones = self.replay_buffer.next_batch (self.batch_size)
 
        target_f = np.zeros((self.batch_size))

        for i in range(self.batch_size):
            if dones[i]:
                target_f[i] = rewards[i]
            else:
                target_f[i] = rewards[i] + self.discount_factor * np.max(self.Q_target.predict(self.sess, [next_states[i]]), 1)
                

        loss = self.Q.update(self.sess, states, actions, target_f)

        self.Q_target.update(self.sess)
        '''
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)
        batch_state, batch_action, batch_next_state, batch_rewards, batch_done = self.replay_buffer.next_batch(self.batch_size)

        td_target =  batch_rewards

        best_action = np.argmax(self.Q.predict(self.sess, batch_next_state)[np.logical_not(batch_done)], 1)
	
        td_target[np.logical_not(batch_done)] += self.discount_factor * self.Q_target.predict(self.sess, batch_next_state)[np.logical_not(batch_done), best_action]

        loss = self.Q.update(self.sess, batch_state, batch_action, td_target)
        self.Q_target.update(self.sess)


        return loss


    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)    
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        '''
        r = np.random.uniform()

        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            # action_id = ...
            action_id = np.argmax(self.Q.predict(self.sess, [state]))
        else:
        
            if self.game == 'cartpole':
                action_id = random.randrange(self.num_actions)
            elif self.game == 'carracing':

            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            # action_id = ...
                
                probabilities = [0.1, 0.2, 0.2, 0.45, 0.05]
                
                action_id = np.random.choice (self.num_actions, p=probabilities)
        '''

        if deterministic:
            
            action_id = np.argmax(self.Q.predict(self.sess, [state]))
        else:
            if self.exploration == "greedy":
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                r = np.random.uniform()
                if r > self.epsilon:
                    # TODO: take greedy action (argmax)
                    
                    action_id = np.argmax(self.Q.predict(self.sess, [state]))
                else:
                    # TODO: sample random action
                    # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work. 
                    # You can sample the agents actions with different probabilities (need to sum up to 1) so that the agent will prefer to accelerate or going straight.
                    # To see how the agent explores, turn the rendering inthe training on and look what the agent is doing.
                    if self.game == "cartpole" :
                        action_id = np.random.randint(self.num_actions)
                    elif self.game == "carracing":
                        probabilities = [0.15, 0.15, 0.15, 0.3, 0.05, 0.1, 0.1]
                
                        action_id = np.random.choice (self.num_actions, p=probabilities)
                    else:
                        print("Invalid game")
            elif self.exploration == "boltzmann":
                action_value = self.Q.predict(self.sess, [state])[0]
                prob = self.softmax(action_value/self.tau)
                action_id = np.random.choice(self.num_actions, p=prob)
            else:
                print("Invalid Exploration Type")


        return action_id


    def softmax(self, input):
        """
        Safe Softmax function to avoid overflow
        Args:
            input: input vector
        Returns:
            prob: softmax of input
        """
        input_max = np.max(input)
        e = np.exp(input-input_max)
        prob = e / np.sum(e)
        return prob


    def load(self, file_name):
        self.saver.restore(self.sess, file_name)


    def check_early_stop(self, reward, totalreward):
        return self.Q_target.check_early_stop (reward, totalreward)
    
