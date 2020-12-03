## This is the implementation of a basic value function approximator (actor)
## Author : Ilyeech Kishore Rapelli
## Date : 20/11/2020

import numpy as np 
from matplotlib import pyplot as plt
from math import pi, sqrt, exp

class LinearFeaturesCritic:

    def __init__(self, env, dt, DEBUG):
        '''
        Input:
            env : the dynamics of the system for which trajectory is generated
            dt : discretization
        '''
        self.env = env
        self.dt = dt
        self.DEBUG = DEBUG
        self.parameters_history = []

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
        self.parameters_history.append(self.parameters)
        self.features_generator = features_generator

    def get_value(self, state):
        state_features = self.features_generator.get_s_features(state)
        state_value = np.dot(self.parameters, state_features)
        if(self.DEBUG):
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
        self.delta = reward - self.get_value(state) + self.gamma* self.get_value(state_new)
        if(self.DEBUG):
            print("Delta: {}".format(self.delta))
    
    def backward_pass(self, state):
        self.parameters = self.parameters + np.multiply(self.get_grad_value(state), self.alpha * self.delta)
        self.parameters_history.append(self.parameters)
        if(self.DEBUG):
            print("Value parameters: {}".format(self.parameters))
        return self.delta

    def plot(self):
        
        plt.plot((180.0/np.pi)*self.x[0], label = "new_traj")
        plt.grid()
        plt.legend()
        plt.show()
        
    def plot_policy(self):
        '''
        This function plots the joint positions, velocities and torques
        '''
        policy_size = len(self.parameters)
        fig, axs = plt.subplots(policy_size,1, figsize = (10, 10))
        self.policy_history = np.asarray(self.parameters_history)
        
        for i in range(policy_size):
            axs[i].plot(self.policy_history[:,i], label = str(i+1)+'st Parameter')
            axs[i].grid()
            axs[i].legend()
            #axs[0].set_ylabel("degrees")
    
    
        plt.show()
        