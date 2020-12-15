## This is the implementation of a basic value function approximator (actor)
## Author : Ilyeech Kishore Rapelli
## Date : 18/11/2020

import numpy as np 
from matplotlib import pyplot as plt
from math import pi, sqrt, exp, radians, degrees

class LinearFeaturesGaussianNACActor:

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
        self.log_gradient_arr = []
        self.episode_cost_arr = []#stores cost after each episode
        self.episode_discounted_cost_arr = []#stores discounted cost after each episode
        self.episode_discounted_log_gradient_arr = []#for each episode
        self.episode_end_position = []
        self.episode_end_angle = []
        self.parameters_history = []

    def initialize(self, T, alpha, gamma, critic, policy_parameters, features_generator, cost, reset, state_init = None):
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
        self.parameters_size = len(policy_parameters)
        self.parameters_history.append(self.parameters)
        self.features_generator = features_generator
        self.cost = cost
        self.factor = 1
        self.torque_limit = 10
        self.cost_arr_index_prev = 0
        self.cost_arr_index = 0##index into all step/forward pass arrays
        self.episode_cost_arr_index_prev = 0
        self.episode_cost_arr_index = 0#index into all episode arrays
        self.reset = reset
        
    def sample_action(self, state):
        state_features = self.features_generator.get_s_features(state)
        if(self.DEBUG):
            print("state features: {}".format(state_features))
        mu = np.dot(self.parameters[0:self.parameters_size - 1], state_features[0:self.parameters_size - 1])
        sigma=1
        #sigma = exp(np.dot(self.parameters[1], state_features))#float?
        #print("log sigma: {}".format(np.dot(self.parameters[1], state_features)))
        action = np.random.normal(mu, sigma)
        if(self.DEBUG):
            print("Mu: {}, Sigma: {}, action: {}".format(mu, sigma, action))
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
        and returns the reward in the step, and if the episode ended
        '''
        t = self.env.t
        end_status = False
        state = self.env.get_state()
        if(self.DEBUG):
            print("State: {}".format(state))
        action, self.grad_log_prob = self.sample_action(state)#will have more dim for 2-DoF & others
        #self.action_history[:,t] = action
        if(self.cost.intermediate_cost != None):
            reward = -1 * (self.cost.intermediate_cost.compute(state, t).item())#the new way to get one item
            if t < 5:
                print(reward)
        if(self.cost.control_cost != None):
            reward += -1 * (self.cost.control_cost.compute(action, t).item())
            if t < 10:
                print(reward)
        if(self.DEBUG):
            print("reward: {}".format(reward))
        self.env.step(float(action), use_euler)
        #jp_new, jp_d_new = self.env.get_joint_state()
        state_new = self.env.get_state()
        if(self.DEBUG):
            print("State shifted to: {}".format(state))
        #self.state_history[:,t+1] = jp_new, jp_d_new#t+1 is the new self.env.t
        #self.critic.forward_pass(state, action, state_new, reward)
        if (self.cost.terminal_cost != None):
            if(self.cost.terminal_cost.end_status(state, self.env.t)):
                reward += -1 * (self.cost.terminal_cost.compute(state, t).item())
                end_status = True
        self.cost_arr.append(float(-reward))
        self.log_gradient_arr.append(self.grad_log_prob)#NEED TO REMOVE THE LAST ITEM for 1
        self.cost_arr_index += 1
        self.old_state = state
        return reward, end_status
    
        # plt.plot((180.0/np.pi)*self.x[0], label = "new_traj")
        # plt.plot((180.0/np.pi)*self.x_nom[0], label = "old_traj")
        # plt.grid()
        # plt.legend()
        # plt.show()
    
    def backward_pass(self):
        #state = np.array(self.env.get_joint_state(), dtype=object)
        delta = self.critic.backward_pass(self.old_state)
        #print(self.parameters)
        self.parameters = self.parameters + self.alpha * self.factor * delta * self.grad_log_prob#backward pass?
        self.parameters_history.append(self.parameters)
        if(self.DEBUG):
            print("Policy parameters: {}".format(self.parameters))
        self.factor *= self.gamma

    def optimize(self, no_iterations = 10, no_episodes = 1, max_episode_length = 100, use_euler = True):
        '''
        This function runs the forward and the backward pass for the policy (gradient) model
        '''
        w = np.random.normal(0., 0.1, self.parameters_size)
        w_old = np.random.normal(0., 0.1, self.parameters_size)
        w_convergence = 0
        episode_success = 0
        for iteration in range(no_iterations):
            self.episode_cost_arr_index_prev = self.episode_cost_arr_index
            actor_updated = 0
            for episode in range(no_episodes):
                if(actor_updated == 1):
                    break
                #Execute rollout
                #self.factor = 1#????
                episode_executed = False
                while episode_executed == False:
                    self.cost_arr_index_prev = self.cost_arr_index
                    self.env.reset_state(self.reset.reset())
                    steps_taken = 0
                    for step in range(max_episode_length):
                        if episode_executed == True:
                            break
                        steps_taken = step
                        reward, episode_executed = self.forward_pass(use_euler)#step level forward pass
                        state = np.array(self.env.get_state(), dtype=object)#ISN'T this already NUMPY?
                        if(step%1 == 0):
                            if(self.DEBUG):
                                print("finished pass {} and the cost is {}".format(step, self.cost_arr[-1]))
                        if(self.DEBUG):
                            print("Policy parameters: {}".format(self.parameters))
                        if(abs(state[0] - 2) > 4 or abs(state[1]) > 1000):
                            if(self.DEBUG):
                                print("Out of bounds. Ending episode " + str(episode) + " in step (index)" + str(step))
                            break
                        if(episode_executed):
                            episode_success += 1
                        if(self.DEBUG and episode_executed == True):
                            print("Episode " + str(episode) + " successfully executed in steps " + str(steps_taken + 1))
                                
                        #print("\n")
                    if(episode_executed == False):
                        reward = reward + -1 * (self.cost.terminal_cost.compute(state, self.env.t))
                        self.cost_arr[-1] = (float(-reward))
                    episode_executed = True
                    
                steps_taken += 1
                self.episode_cost_arr_index += 1
                
                #Episode level forward pass
                episode_cost = sum(self.cost_arr[self.cost_arr_index_prev: self.cost_arr_index])
                #assertion failed some times only. steps_taken was increased even after episode was forced ended. 
                #assert(steps_taken == self.cost_arr_index - self.cost_arr_index_prev)
                episode_discounted_cost = np.dot(np.power(self.gamma, np.array(range(steps_taken))),
                                                 np.array(self.cost_arr[self.cost_arr_index_prev: self.cost_arr_index]))
                episode_discounted_log_gradient = np.dot(np.power(self.gamma, np.array(range(steps_taken))),
                                                 np.array(self.log_gradient_arr[self.cost_arr_index_prev: self.cost_arr_index]))
                self.episode_cost_arr.append(episode_cost)
                self.episode_end_position.append(state[0])
                self.episode_discounted_cost_arr.append(episode_discounted_cost)
                self.episode_discounted_log_gradient_arr.append(episode_discounted_log_gradient)
                
                #Backward pass (conditional)
                if(episode >= self.parameters_size - 1):
                    if(episode == self.parameters_size):#if no of episodes is (dim theta) + 1
                        w_old = np.random.normal(0., 0.1, self.parameters_size)
                    else:
                        w_old = w
                    #print("Episode Cost array indices: " + str(self.episode_cost_arr_index_prev) + " , " + str(self.episode_cost_arr_index))
                    episodes_in_iteration = self.episode_cost_arr_index - self.episode_cost_arr_index_prev
                    #Can attempt solving the linear equations for the natural gradient w
                    basis_functions = np.array(self.episode_discounted_log_gradient_arr[
                        self.episode_cost_arr_index_prev:self.episode_cost_arr_index])
                    #print("Basis functions: " + str(basis_functions))
                    basis_functions[:, self.parameters_size - 1] = np.full(episodes_in_iteration, 1.0)#the last column has to be 1
                    reward_statistics = -1 * np.array(self.episode_discounted_cost_arr[
                        self.episode_cost_arr_index_prev:self.episode_cost_arr_index])
                    #print("Basis functions: " + str(basis_functions))
                    #print("reward statistics: " + str(reward_statistics))
                    
                    #The update to the critic
                    #w = np.linalg.solve(basis_functions, reward_statistics)
                    #For over-determined systems, linalg.solve doesn't use pseudo inverse automatically.
                    #Need to use pinv()
                    #https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.9-The-Moore-Penrose-Pseudoinverse/
                    basis_functions_moore_penrose_pseudoinverse = np.linalg.pinv(basis_functions)
                    w = np.dot(basis_functions_moore_penrose_pseudoinverse, reward_statistics)
                    if(self.DEBUG > 0):
                        print("w: " + str(w))
                    w_trimmed = w[0:self.parameters_size - 1]#Cutting out J
                    #print("w_trimmed: " + str(w_trimmed))
                    w_old_trimmed = w_old[0:self.parameters_size - 1]
                    cosine_similarity = np.dot(w_trimmed, w_old_trimmed)/(np.linalg.norm(w_trimmed) * np.linalg.norm(w_old_trimmed))
                    if(cosine_similarity > 0.9998 and episode > self.parameters_size):
                        #cos(1) = 0.9998, cos(2) = 0.9994, cos(5) = 0.9962, cos(10) = 0.9848
                        w_convergence += 1
                        if(iteration%50 == 0):
                            print("In iteration " + str(iteration) +", w converged in " + str(episode + 1) + " episodes ")
                            print("J = " + str(w[self.parameters_size-1]))
                        #The update to the actor (policy)
                        self.parameters = self.parameters + self.alpha * w
                        self.parameters_history.append(self.parameters)
                        if(self.DEBUG):
                            print("Policy parameters: {}".format(self.parameters))
                        actor_updated = 1
                    else:
                        #more episodes.
                        pass
        print("\nEpisode succeeded", episode_success, "times out of", self.episode_cost_arr_index, "episodes")
        print("w converged " + str(w_convergence) + " times out of " + str(no_iterations) + " iterations")
                   
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
        
    def plot_end_position(self):
        x = np.linspace(0, len(self.episode_end_position) - 1, len(self.episode_end_position))
        plt.scatter(x, self.episode_end_position, label = "End position of Episode")
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