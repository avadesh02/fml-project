## This file contains the implementation of cost functions
## Author : Avadesh Meduri
## Date : 9/11/2020

import numpy as np 

class PositionTrackingCost:

    def __init__(self, env, x_nom, Q, istimeinvariant = True):
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
        self.x_nom = x_nom

        self.istimeinvariant = istimeinvariant # false if it changes with time

    def x(self, state, t):
        '''
        computes the derivative of the cost with respect to state
        Input:
            state : state at time t
            t : time
        '''
        if self.istimeinvariant:
            return np.matmul(self.Q, np.transpose(state - self.x_nom))
        else:
            return np.matmul(self.Q[t], np.transpose(state - self.x_nom[t]))
            
    def xx(self, state, t):
        '''
        returns second derivative of cost wrt state
        Input:
            state : state fo the system at time t
            t : time
        '''
        return self.Q


class TerminalPositionTrackingCost:

    def __init__(self, env, x_des, Q):
        '''
        This the terminal cost to track desired positions x^T(Q)x
        Input:
            env : dynamics of the system
            x_des : final desired state
            Q : weight matrix
        '''
        self.env = env
        assert np.shape(Q)[0] == env.no_states
        self.Q = Q
        self.x_des = x_des
        
    def x(self, state, t):
        '''
        computes the derivative of the cost with respect to state
        Input:
            state : state at time t
            t : time
        '''

        return np.matmul(self.Q[t], np.transpose(state - self.x_des))
            
    def xx(self, state, t):
        '''
        returns second derivative of cost wrt state
        Input:
            state : state fo the system at time t
            t : time
        '''
        return self.Q


class ControlRegularizerCost:

    def __init__(self, env, R, istimeinvariant = True):
        '''
        Regularizing cost on control u^T(R)u
        '''
        self.env = env
        assert np.shape(R)[0] == self.env.no_actions
        self.R = R
        self.istimeinvariant = istimeinvariant

    def u(self, action, t):
        '''
        computes first derivative of cost wrt u
        Input:
            action : action at time t
            t : time
        '''

        if self.istimeinvariant:
            return np.matmul(self.R, np.transpose(action))
        else:
            return np.matmul(self.R[t], np.transpose[action])
    
    def uu(self, action, t):
        '''
        computes second derivative of cost wrt u
        Input:
            action : action at time t
            t : time
        '''
        return self.R