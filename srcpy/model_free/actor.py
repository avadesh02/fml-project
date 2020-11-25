## This is the implementation of a basic value function approximator (actor)
## Author : Ilyeech Kishore Rapelli
## Date : 18/11/2020

import numpy as np 
from matplotlib import pyplot as plt
from math import pi, sqrt, exp

class LinearFeaturesGaussianActor:

    def __init__(self, env, dt):
        '''
        Input:
            env : the dynamics of the system for which trajectory is generated
            dt : discretization
        '''
        self.env = env
        self.dt = dt
        self.gradient = [] # stores gradient after each forward pass. Not using as on 21 Nov
        self.cost_arr = [] # stores cost after each forward pass

    def initialize(self, T, alpha, gamma, critic, policy_parameters, features_generator, cost, state_init = None):
        '''
        This function intialises the variables for the actor
        Input:
            alpha: learning rate for policy parameters
            critic: the critic object (initialized according to the model)
        '''
        self.n = int(np.round(T/self.dt, 1))
        self.state_history = np.zeros((self.env.no_states, self.n + 1))
        self.action_history = np.zeros((self.env.no_actions, self.n))
        self.alpha = alpha
        self.gamma = gamma
        self.critic = critic
        assert(self.gamma == critic.gamma)
        self.parameters = policy_parameters
        self.features_generator = features_generator
        self.cost = cost
        self.factor = 1
        self.torque_limit = 30
        
    def sample_action(self, state):
        state_features = self.features_generator.get_s_features(state)
        print("state features: {}".format(state_features))
        #print(self.parameters[0:4])
        #print(self.parameters[4:8])
        mu = np.dot(self.parameters[0:4], state_features)
        sigma = exp(np.dot(self.parameters[4:8], state_features))#float?
        print("Mu: {}, Sigma: {}".format(mu, sigma))
        action = np.random.normal(mu, sigma)
        action = max(-self.torque_limit, min(action, self.torque_limit))
        print("Torque: {}".format(action))
        state_action_features = self.features_generator.get_sa_features(state, action)
        print("S-A Features: {}".format(state_action_features))
        # Sigma going down to 0 is a real problem in stochastic policies it seems.
        #epsilon can be a function of I, l
        if(abs(sigma) <  0.0001):
            grad_log_prob = mu
        else:
            grad_log_prob = np.multiply(state_action_features, 
                                    (1/(sigma ** 2) * (action - mu)) )
        #print("Grad log prob: {}".format(grad_log_prob))
        return action, grad_log_prob

    def forward_pass(self):
        '''
        This function runs the forward pass for the actor
        '''
        t = self.env.t
        jp, jp_d = self.env.get_joint_state()
        state = np.array([jp, jp_d], dtype=object)#line can be merged with line above.
        print("State: {}".format(state))
        action, self.grad_log_prob = self.sample_action(state)#will have more dim for 2-DoF & others
        self.action_history[:,t] = action
        reward = -1 * (self.cost.compute(state, t).item())#the new way to get one item
        print("reward: {}".format(reward))
        self.env.step_manipulator(float(action), use_euler = True)
        jp_new, jp_d_new = self.env.get_joint_state()
        state_new = np.array([jp_new, jp_d_new], dtype=object)
        self.state_history[:,t+1] = jp_new, jp_d_new
        self.critic.forward_pass(state, action, state_new, reward)
        self.cost_arr.append(float(-reward))
    
        # plt.plot((180.0/np.pi)*self.x[0], label = "new_traj")
        # plt.plot((180.0/np.pi)*self.x_nom[0], label = "old_traj")
        # plt.grid()
        # plt.legend()
        # plt.show()
    
    def backward_pass(self):
        state = np.array(self.state_history[:,self.env.t], dtype=object)
        delta = self.critic.backward_pass(state)
        print("Delta: {}".format(delta))
        #print(self.parameters)
        self.parameters = self.parameters + self.alpha * self.factor * delta * self.grad_log_prob#backward pass?
        print("Policy parameters: {}".format(self.parameters))
        self.factor *= self.gamma

    def optimize(self, no_iterations = 10):
        '''
        This function runs the forward and the backward pass for the policy (gradient) model
        '''
        for n in range(no_iterations):
            self.forward_pass()
            self.backward_pass()
            if(n%1 == 0):
                print("finished pass {} and the cost is {}\n".format(n, self.cost_arr[-1]))

    def plot(self):  
        plt.plot((180.0/np.pi)*self.state_history[0], label = "traj")
        plt.grid()
        plt.legend()
        plt.show()
        