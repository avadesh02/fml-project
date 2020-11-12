## This is the implementation of the iterative linear quadratic regulator
## algorithms
## Author : Avadesh Meduri
## Date : 9/11/2020

import numpy as np 
from matplotlib import pyplot as plt 

class ILQR:

    def __init__(self, env, dt):
        '''
        Input:
            env : the dynamics of the system for which trajectory is generated
            dt : discretization
        '''
        self.env = env
        self.dt = dt
        self.cost_arr = [] # stores cost after each forward pass

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
        self.n = int(np.round(T/self.dt, 1))
        self.x_nom = np.zeros((self.env.no_states, self.n + 1)) # trajectory around which linearization is done
        self.x = np.zeros((self.env.no_states, self.n + 1))
        self.u = np.zeros((self.env.no_actions, self.n))
        self.K = np.zeros((self.env.no_actions, self.env.no_states, self.n)) # feedback
        self.k = np.zeros((self.env.no_actions, self.n)) ## feed forward

        if np.array(x_init).any():
            self.x_nom[:,0] = x_init
            self.x[:,0] = x_init

        if np.array(u_init).any():
            self.u = u_init

        self.Q_t = Q_t
        self.R_t = R_t
        self.Q_f = Q_f

    def forward_pass(self):
        '''
        This function runs the forward pass for the ilqr
        '''
        cost = 0
        for t in range(self.n):
            self.u[:,t] += np.matmul(self.K[:,:,t], (self.x[:,t] - self.x_nom[:,t])) + self.k[:,t]
            self.x[:,t+1] = self.env.integrate_dynamics(self.x[:,t], self.u[:,t])
            cost += self.Q_t.compute(self.x[:,t], t) + self.R_t.compute(self.u[:,t],t)

        cost += self.Q_f.compute(self.x[:,t+1])
        self.cost_arr.append(float(cost))
    
        # plt.plot((180.0/np.pi)*self.x[0], label = "new_traj")
        # plt.plot((180.0/np.pi)*self.x_nom[0], label = "old_traj")
        # plt.grid()
        # plt.legend()
        # plt.show()

        self.x_nom = self.x.copy()
        

    def backward_pass(self):
        for t in range(self.n - 1, 0, -1):
            if t == self.n - 1:
                V_xx = self.Q_f.xx(self.x[:,t+1])
                V_x = self.Q_f.x(self.x[:,t+1])
            
            ## assuming l_ux = 0
            l_ux = 0
            l_x = self.Q_t.x(self.x_nom[:,t], t)
            l_xx = self.Q_t.xx(self.x_nom[:,t], t)
            l_u = self.R_t.u(self.u[:,t], t)
            l_uu = self.R_t.uu(self.u[:,t], t)

            f_x = self.env.dynamics_x(self.x_nom[:,t], self.u[:,t])
            f_u = self.env.dynamics_u(self.x_nom[:,t], self.u[:,t])

            Q_x = l_x + np.matmul(np.transpose(f_x), V_x)
            Q_u = l_u + np.matmul(np.transpose(f_u), V_x)
            Q_xx = l_xx + np.matmul(np.matmul(np.transpose(f_x), V_xx), f_x)
            Q_ux = np.matmul(np.matmul(np.transpose(f_u), V_xx), f_x)
            Q_uu = l_uu + np.matmul(np.matmul(np.transpose(f_u), V_xx), f_u)
            self.K[:,:,t] = -np.matmul(np.linalg.inv(Q_uu),np.matrix(Q_ux))
            self.k[:,t] = -np.matmul(np.linalg.inv(Q_uu),np.matrix(Q_u))

            V_x = Q_x - np.matmul(np.matmul(np.matrix(self.K[:,:,t]).transpose(), Q_uu), np.matrix(self.k[:,t]))
            V_xx = Q_xx - np.matmul(np.matmul(np.matrix(self.K[:,:,t]).transpose(), Q_uu), self.K[:,:,t])

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
        