## This file contains terminal cost functions
## Author : Avadesh Meduri
## Date : 12/11/2020

import numpy as np 

class TerminalQuadraticTrackingCost:

    def __init__(self, env, x_nom, Q):
        '''
        This the running cost to track desired positions x^T(Q)x
        Input:
            env : dynamics of the system
            x_nom : nominal desired trajecotory 
            Q : cost matrix
        '''
        self.env = env
        assert np.shape(Q)[0] == env.no_states
        self.Q = Q
        assert len(x_nom) == env.no_states
        self.x_nom = x_nom

    def compute(self, state):
        '''
        This function computes the cost at time t 
        Input:
            state : state at time t
        '''
        return 0.5*np.matmul(np.matmul((state - self.x_nom), self.Q), np.matrix(state - self.x_nom).transpose()) 
            
    def x(self, state):
        '''
        computes the derivative of the cost with respect to state
        Input:
            state : state at time t
        '''
        return np.matmul(self.Q, np.matrix(state - self.x_nom).transpose())
        
    def u(self, state):
        '''
        computes derivative wrt to action
        '''
        return np.zeros((self.env.no_actions, 1))

    def ux(self, state):
        '''
        computes ux
        '''
        return np.zeros((self.env.no_actions, self.env.no_states))

    def xx(self, state):
        '''
        returns second derivative of cost wrt state
        Input:
            state : state fo the system at time t
            t : time
        '''
        return self.Q
        
    def uu(self, state):
        '''
        computes second derivative wrt to u
        '''

        return np.zeros((self.env.no_actions, self.env.no_actions))
