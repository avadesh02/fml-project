## This file contains the implementation of cost functions
## Author : Ilyeech Kishore Rapelli
## Date : 20/11/2020

import numpy as np 

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
        #if self.istimeinvariant:
        return 0.5*np.matmul(np.matmul((state - self.x_nom), self.Q), np.matrix(state - self.x_nom).transpose())