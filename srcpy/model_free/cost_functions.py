## This file contains the implementation of cost functions
## Author : Ilyeech Kishore Rapelli
## Date : 20/11/2020

import numpy as np 

class Cost:
    def __init__(self):
        pass
    
    def initialize(self, terminal_cost = None, intermediate_cost = None, control_cost = None):
        self.terminal_cost = terminal_cost
        self.intermediate_cost = intermediate_cost
        self.control_cost = control_cost
        

class QuadraticCost:

    def __init__(self, env, x_nom, Q, istimeinvariant = True):
        '''
        Input:
            env : dynamics of the system
        '''
        self.env = env
        assert np.shape(Q)[0] == env.no_states
        self.Q = Q
        assert len(x_nom) == env.no_states
        self.x_nom = x_nom
        self.istimeinvariant = istimeinvariant # false if it changes with time

    def compute(self, state, t):
        '''
        This function computes the cost at time t 
        Input:
            state : state at time t
            t : time
        '''
        #if(abs(state[0] - self.x_nom[0]) < 0.001 and abs(state[1] - self.x_nom[1]) < 0.001):
         #   return -100000
        #if self.istimeinvariant:
        return 0.5*np.matmul(np.matmul((state - self.x_nom), self.Q), np.matrix(state - self.x_nom).transpose())
    
class QuadraticTerminalCost:

    def __init__(self, env, x_nom, Q, istimeinvariant = True):
        '''
        Input:
            env : dynamics of the system
        '''
        self.env = env
        assert np.shape(Q)[0] == env.no_states
        self.Q = Q
        assert len(x_nom) == env.no_states
        self.x_nom = x_nom
        self.istimeinvariant = istimeinvariant # false if it changes with time

    def compute(self, state, t):
        '''
        This function computes the cost at time t 
        Input:
            state : state at time t
            t : time
        '''
        #if(abs(state[0] - self.x_nom[0]) < 0.001 and abs(state[1] - self.x_nom[1]) < 0.001):
         #   return -100000
        #if self.istimeinvariant:
        if(abs(state[0] - self.x_nom[0]) < 0.1 and abs(state[1] - self.x_nom[1]) < 0.1):
            print("\n\n\nYay, reached the terminal state\n\n\n")
            return -10000
        else:
            return 0

class ControlRegularizerCost:

    def __init__(self, env, R, istimeinvariant = True):
        '''
        Regularizing cost on control u^T(R)u
        '''
        self.env = env
        if self.env.no_actions > 1:
            assert np.shape(R)[0] == self.env.no_actions
        self.R = R
        self.istimeinvariant = istimeinvariant

    def compute(self, action, t):
        '''
        This function computes the cost at time t 
        Input:
            action : action at time t
            t : time
        '''
        if self.istimeinvariant:
            return 0.5*np.matmul(np.matmul(np.matrix(action), np.matrix(self.R)), np.matrix(action).transpose()) 
        else:
            return 0.5*np.matmul(np.matmul(np.matrix(action), np.matrix(self.R[t])), np.matrix(action).transpose()) 