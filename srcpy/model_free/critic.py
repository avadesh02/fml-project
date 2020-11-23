## This is the implementation of a basic value function approximator (actor)
## Author : Ilyeech Kishore Rapelli
## Date : 20/11/2020

import numpy as np 
from matplotlib import pyplot as plt
from math import pi, sqrt, exp

class LinearFeaturesCritic:

    def __init__(self, env, dt):
        '''
        Input:
            env : the dynamics of the system for which trajectory is generated
            dt : discretization
        '''
        self.env = env
        self.dt = dt
        self.gradient = [] # stores gradient after each forward pass

    def initialize(self, alpha, gamma, parameters, features_generator):
        '''
        This function intialises the variables for the actor
        Input:
            alpha: learning rate for policy parameters
            critic: the critic object (initialized according to the model)
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.parameters = parameters
        self.features_generator = features_generator

    def get_value(self, state):
        state_features = self.features_generator.get_s_features(state)
        state_value = np.dot(self.parameters[0:4], state_features)
        print("State value: {}".format(state_value))
        return float(state_value)
    
    def get_grad_value(self, state):
        state_features = self.features_generator.get_s_features(state)
        return state_features#for linear features, features = grad

    def forward_pass(self, state, action, state_new, reward):#env is available. arguments necessary?
        '''
        This function runs the forward pass for the critic
        '''
        #self.gradient = []
        self.delta = reward + self.gamma* self.get_value(state_new) - self.get_value(state)
    
    def backward_pass(self, state):
        self.parameters = self.parameters + np.multiply(self.get_grad_value(state), self.alpha * self.delta)
        return self.delta
        
    

    def optimize(self, no_iterations = 10):
        '''
        This function runs ilqr and returs optimal trajectory
        '''
        for n in range(no_iterations):
            self.forward_pass()
            self.backward_pass()
            print("finished iteration {} and the cost is {}".format(n, self.cost_arr[-1]))
        
        return self.x_nom, self.K, self.u

    def plot(self):
        
        plt.plot((180.0/np.pi)*self.x[0], label = "new_traj")
        plt.grid()
        plt.legend()
        plt.show()
        