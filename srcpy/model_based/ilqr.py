## This is the implementation of the iterative linear quadratic regulator
## algorithms
## Author : Avadesh Meduri
## Date : 9/11/2020

import numpy as np 

class ILQR:

    def __init__(self, env, dt):
        '''
        Input:
            env : the dynamics of the system for which trajectory is generated
            dt : discretization
        '''
        self.env = env
        self.dt = dt

    def initialize(self, T, Q_t, R_t, Q_f, x_init = None, u_init = None):
        '''
        This function intialises the variables for the optimization
        Input:
            x_init : initial state of system
            u_init : initial trajectory of system
            Q_T : running cost matrix class for state
            R_t : running cost matrx class for control
            Q_f : teminal cost
        '''
        self.n = np.round(T/self.dt, 1)
        self.x_nom = np.zeros((self.env.no_states, self.n + 1)) # trajectory around which linearization is done
        self.x = np.zeros((self.env.no_states, self.n + 1))
        self.u = np.zeros((self.env.no_actions, self.n))
        self.K = np.zeros((self.env.no_actions, self.env.no_states, self.n)) # feedback
        self.k = np.zeros((self.env.no_actions, self.n)) ## feed forward

        if x_init:
            self.x_nom[:,0] = x_init
            self.x[:,0] = x_init

        if u_init:
            self.u = u_init

        self.Q_t = Q_t
        self.R_t = R_t
        self.Q_f = Q_f

    def forward_pass(self):
        '''
        This function runs the forward pass for the ilqr
        '''
        for t in range(self.n):
            self.u[:,t] -= np.matmul(self.K[::,t], (self.x[:,t] - self.x_nom[:,t])) + self.k[:,t]
            self.x[:,t+1] = self.env.integrate_dyamics(self.x[:,t], self.u[:,t])

        self.x_nom = self.x.copy()

    def backward_pass(self):
        '''
        This function runs the backward pass for the ilqr
        '''

        for t in range(self.n, 0, -1):
            if t == self.n - 1:
                S_t = self.Q_f.xx(self.x[:,t+1])
                v_t = self.Q_f.x(self.x[:,t+1])
            
            f_x = self.env.dynamics_x(self.x_nom[:,t], self.u[:,t])
            f_u = self.env.dynamics_u(self.x_nom[:,t], self.u[:,t])
            
            tmp = np.linalg.inv(np.matmul(np.matmul(np.transpose(f_u),S_t),f_u + self.R_t.uu(self.u[:,t])))
            
            self.K[::,t] = np.matmul(np.matmul(np.matmul(tmp,np.transpose(f_u)),S_t),f_x)
            self.k[:,t] = np.matmul(tmp, np.matmul(np.transpose(f_u), v_t) + np.matmul(self.R_t.uu[self.u[:,t]], self.u[:,t]))
            
            S_t = np.matmul(np.matmul(f_x.transpose(), S_t), f_x - np.matmul(f_u, self.K[::,t])) + self.Q_t.xx(self.x_nom[:,t], t) 
            
            v_t = np.matmul(np.transpose((f_x - np.matmul(f_u, self.K[::,t]))),v_t)
            v_t -= np.matmul(np.matmul(np.transpose(self.K[::,t]),self.R_t.uu(self.u[:,t], t)),self.u[:,t])
            v_t += np.matmul(self.Q_t.xx(self.x_nom[:,t], t), self.x_nom[:,t])

    def optimize(self, no_iterations = 10):
        '''
        This function runs ilqr and returs optimal trajectory
        '''
        