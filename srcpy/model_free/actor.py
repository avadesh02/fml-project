## This is the implementation of a basic value function approximator (actor)
## Author : Ilyeech Kishore Rapelli
## Date : 18/11/2020

import numpy as np 
from matplotlib import pyplot as plt
from math import pi, sqrt, exp, radians, degrees

class LinearFeaturesGaussianActor:

    def __init__(self, env, dt, DEBUG):
        '''
        Input:
            env : the dynamics of the system for which trajectory is generated
            dt : discretization
        '''
        self.env = env
        self.dt = dt
        self.DEBUG = DEBUG
        self.cost_arr = [] # stores cost after each forward pass
        self.episode_cost_arr = []#stores cost after each episode
        self.parameters_history = []

    def initialize(self, T, alpha, gamma, critic, policy_parameters, features_generator, cost, state_init = None):
        '''
        This function intialises the variables for the actor
        Input:
            alpha: learning rate for policy parameters
            critic: the critic object (initialized according to the model)
        '''
        self.n = int(np.round(T/self.dt, 1))
        self.state_history = np.zeros((self.env.no_states, self.n + 2))
        self.state_history[0][0] = state_init[0]
        self.state_history[1][0] = state_init[1]
        self.action_history = np.zeros((self.env.no_actions, self.n + 1))#Shouldn't need +1
        self.action_history[0] = 0.
        self.alpha = alpha
        self.gamma = gamma
        self.critic = critic
        assert(self.gamma == critic.gamma)
        self.parameters = policy_parameters
        self.parameters_history.append(self.parameters)
        self.features_generator = features_generator
        self.cost = cost
        self.factor = 1
        self.torque_limit = 10
        self.cost_arr_index_prev = 0
        self.cost_arr_index = 0
        
    def sample_action(self, state):
        state_features = self.features_generator.get_s_features(state)
        if(self.DEBUG):
            print("state features: {}".format(state_features))
        mu = np.dot(self.parameters, state_features)
        sigma=0.5
        #sigma = exp(np.dot(self.parameters[1], state_features))#float?
        #print("log sigma: {}".format(np.dot(self.parameters[1], state_features)))
        if(self.DEBUG):
            print("Mu: {}, Sigma: {}".format(mu, sigma))
        action = np.random.normal(mu, sigma)
        #action = max(-self.torque_limit, min(action, self.torque_limit))#clipping
        if(self.DEBUG):
            print("Torque: {}".format(action))
        #state_action_features = self.features_generator.get_sa_features(state, action)
        #print("S-A Features: {}".format(state_action_features))
        # Sigma going down to 0 is a real problem in stochastic policies it seems.
        #epsilon can be a function of I, l
        if(abs(sigma) <  0.0001):#clipping of sorts
            pass #grad_log_prob = mu #bug this one
        else:
            grad_log_prob = np.multiply(state_features, 
                                    (1/(sigma ** 2) * (action - mu)) )
        if(self.DEBUG):
            print("Grad log prob: {}".format(grad_log_prob))
        return action, grad_log_prob

    def forward_pass(self, use_euler = True):
        '''
        This function runs the forward pass for the actor
        '''
        t = self.env.t
        #jp, jp_d = self.env.get_joint_state()
        jp, jp_d = self.env.get_state()
        state = np.array([jp, jp_d], dtype=object)#line can be merged with line above.
        if(self.DEBUG):
            print("State: {}".format(state))
        action, self.grad_log_prob = self.sample_action(state)#will have more dim for 2-DoF & others
        #self.action_history[:,t] = action
        reward = -1 * (self.cost.intermediate_cost.compute(state, t).item())#the new way to get one item
        if(self.cost.control_cost != None):
            reward += -1 * (self.cost.control_cost.compute(action, t).item())
        if(self.cost.terminal_cost != None):
            reward += -1 * (self.cost.terminal_cost.compute(state, t))
        if(self.DEBUG):
            print("reward: {}".format(reward))
        #self.env.step_manipulator(float(action), use_euler = use_euler)
        self.env.step(float(action))
        #jp_new, jp_d_new = self.env.get_joint_state()
        jp_new, jp_d_new = self.env.get_state()
        state_new = np.array([jp_new, jp_d_new], dtype=object)
        #print("t: {}, self.env.t: {}".format(t, self.env.t))
        #self.state_history[:,t+1] = jp_new, jp_d_new#t+1 is the new self.env.t
        self.critic.forward_pass(state, action, state_new, reward)
        self.cost_arr.append(float(-reward))
        self.cost_arr_index += 1
        self.old_state = state
        return reward
    
        # plt.plot((180.0/np.pi)*self.x[0], label = "new_traj")
        # plt.plot((180.0/np.pi)*self.x_nom[0], label = "old_traj")
        # plt.grid()
        # plt.legend()
        # plt.show()
    
    def backward_pass(self):
        delta = self.critic.backward_pass(self.old_state)
        self.parameters = self.parameters + self.alpha * self.factor * delta * self.grad_log_prob#backward pass?
        self.parameters_history.append(self.parameters)
        if(self.DEBUG):
            print("Policy parameters: {}".format(self.parameters))
        self.factor *= self.gamma

    def optimize(self, max_episode_length = 10, no_episodes = 1, use_euler = True):
        '''
        This function runs the forward and the backward pass for the policy (gradient) model
        '''
        for episode in range(no_episodes):
            #self.factor = 1#????
            self.cost_arr_index_prev = self.cost_arr_index
            new_init_theta = 0#name has to change
            new_init_vel = 0
            #new_init_theta = np.random.uniform(radians(89), radians(91))
            #new_init_vel = np.random.uniform(-radians(2.),radians(2.))
            self.env.reset_state(new_init_theta, new_init_vel)
            iter = 0
            for n in range(max_episode_length):
                iter = n
                self.forward_pass(use_euler)
                #reward = self.forward_pass()
                self.backward_pass()
                if(n%1 == 0):
                    if(self.DEBUG):
                        print("finished pass {} and the cost is {}\n".format(n, self.cost_arr[-1]))
                #if(abs(reward) < 0.001):
                 #   print("\nBREAKING 'cuz target reached\n")
                  #  break
                #state = np.array(self.env.get_joint_state(), dtype=object)
                state = np.array(self.env.get_state(), dtype=object)
                #if(abs(state[0] - radians(90)) > radians(2) or abs(state[1]) > 100):
                if(abs(state[0] - 2) > 4 or abs(state[1]) > 1000):
                    print("Out of bounds. Ending step " + str(n) + " in episode " + str(episode))
                    break
            episode_cost = sum(self.cost_arr[self.cost_arr_index_prev: self.cost_arr_index])
            self.episode_cost_arr.append(episode_cost)
            if(episode %1 == 0):
                #print("Episode {} ({}, {})\t: done with cost {} \tin iterations {}.".format(episode, new_init_theta - radians(90), new_init_vel, episode_cost, iter+1))
                print("Episode {} ({}, {}): done with cost {} \tin steps {}.".format(episode, new_init_theta, new_init_vel, episode_cost, iter+1))
    def plot(self):  
        plt.plot((180.0/np.pi)*self.state_history[0], label = "trajectory")
        plt.grid()
        plt.legend()
        plt.show()
        
    def plot_vel(self):  
        plt.plot((180.0/np.pi)*self.state_history[1], label = "velocity")
        plt.grid()
        plt.legend()
        plt.show()
        
    def plot_torque(self):  
        plt.plot(self.action_history[0], label = "torque")
        plt.grid()
        plt.legend()
        plt.show()
        
    def plot_episode_cost(self, alpha = 0.99):  
        moving_average_cost = self.episode_cost_arr.copy()
        for i in range(len(self.episode_cost_arr) - 1):
            moving_average_cost[i + 1] = alpha * moving_average_cost[i] + (1. - alpha) * self.episode_cost_arr[i+1]
        plt.plot(moving_average_cost, label = "Moving Avg of Episode Cost")
        plt.grid()
        plt.legend()
        plt.show()
    
    def plot_policy(self):
        '''
        This function plots the joint positions, velocities and torques
        '''
        policy_size = len(self.parameters)
        fig, axs = plt.subplots(policy_size,1, figsize = (10, 10))
        self.policy_history = np.asarray(self.parameters_history)#Maybe, it can all be done in numpy arrays
            
        for i in range(policy_size):
            axs[i].plot(self.policy_history[:,i], label = str(i+1)+'st Parameter')
            axs[i].grid()
            axs[i].legend()
            #axs[0].set_ylabel("degrees")
    
    
        plt.show()
    
    def plot_policy_unused(self):
        '''
        This function plots the joint positions, velocities and torques
        '''
        policy_size = len(self.parameters)
        fig, axs = plt.subplots(policy_size,1, figsize = (10, 10))
        self.policy_history = np.asarray(self.parameters_history)#Maybe, it can all be done in numpy arrays
            
        axs[0].plot(self.policy_history[:,0], label = '1st Parameter')
        axs[0].grid()
        axs[0].legend()
        #axs[0].set_ylabel("degrees")

        axs[1].plot(self.policy_history[:,1], label = '2nd Parameter')
        axs[1].grid()
        axs[1].legend()
        #axs[1].set_ylabel("degrees/sec")
    
        axs[2].plot(self.policy_history[:,2], label = '3rd Parameter')
        axs[2].grid()
        axs[2].legend()
        #axs[2].set_ylabel("Newton/(Meter Second)")
            
        axs[3].plot(self.policy_history[:,3], label = '4th Parameter')
        axs[3].grid()
        axs[3].legend()
        #axs[0].set_ylabel("degrees")

        axs[4].plot(self.policy_history[:,4], label = '5th Parameter')
        axs[4].grid()
        axs[4].legend()
        #axs[1].set_ylabel("degrees/sec")
    
    
        plt.show()