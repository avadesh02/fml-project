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
        self.iter_cost_arr = [] # stores cost after each forward pass
        self.iter_traj_arr = [] # stores new trajectory after every iteration

    def initialize(self, T, x_init = None, u_init = None):
        '''
        This function intialises the variables for the optimization
        Input:
            x_init : initial state of system
            u_init : initial trajectory of system
        '''
        self.n = int(np.round(T/self.dt, 1))
        self.x_nom = np.zeros((self.env.no_states, self.n + 1)) # trajectory around which linearization is done
        self.x = np.zeros((self.env.no_states, self.n + 1))
        self.u = np.zeros((self.env.no_actions, self.n))
        self.K = np.zeros(( self.n, self.env.no_actions, self.env.no_states)) # feedback
        self.k = np.zeros((self.n, self.env.no_actions)) ## feed forward

        if np.array(x_init).any():
            self.x_nom[:,0] = x_init
            self.x[:,0] = x_init

        if np.array(u_init).any():
            self.u = u_init

        self.running_cost_arr = []
        self.terminal_cost_arr = []

    def add_running_cost(self, cost):
        '''
        This function appends the running cost for the ilqr problem
        Input:
            cost : cost class which computes derivates etc...
        '''
        self.running_cost_arr.append(cost)

    def add_terminal_cost(self, cost):
        '''
        This function appends the terminal cost for the ilqr problem
        Input:
            cost : cost class which computes derivates etc...
        '''
        self.terminal_cost_arr.append(cost)


    def compute_running_cost(self, t):
        '''
        This computes the value of the cost at time t for the given state and action
        '''
        cost = 0
        for cost_func in self.running_cost_arr:
            cost += cost_func.compute(self.x[:,t].copy(), self.u[:,t].copy(), t)
        return cost

    def compute_terminal_cost(self, t):
        '''
        This computes the value of the terminal cost for the given state and action
        '''
        cost = 0
        for cost_func in self.terminal_cost_arr:
            cost += cost_func.compute(self.x[:,t].copy())
        
        return cost

    def compute_running_cost_derivatives(self, state, action, t):
        '''
        computes all the derivates of the running cost
        Input:
            state : state of the system at time t
            action : action at time t
            t : time
        '''
        l_x = np.zeros((self.env.no_states, 1))
        l_xx = np.zeros((self.env.no_states, self.env.no_states))
        l_u = np.zeros((self.env.no_actions, 1))
        l_ux = np.zeros((self.env.no_actions, self.env.no_states))
        l_uu = np.zeros((self.env.no_actions, self.env.no_actions))

        for cost_func in self.running_cost_arr:
            l_x += cost_func.x(state, action, t)
            l_xx += cost_func.xx(state, action, t)
            l_u += cost_func.u(state, action, t)
            l_ux += cost_func.ux(state, action, t)
            l_uu += cost_func.uu(state, action, t)

        return l_x, l_u, l_ux, l_xx, l_uu

    def compute_terminal_cost_derivates(self, state):
        '''
        computes all the derivatives of the terminal cost
        Input:
            state : state of the system at time t
            action : action at time t
            t : time
        '''
        l_x = np.zeros((self.env.no_states, 1))
        l_xx = np.zeros((self.env.no_states, self.env.no_states))
        for cost_func in self.terminal_cost_arr:
            l_x += cost_func.x(state)
            l_xx += cost_func.xx(state)

        return l_x, l_xx

    def forward_pass(self):
        '''
        This function runs the forward pass for the ilqr
        '''
        epi_cost = 0
        for t in range(self.n):
            self.u[:,t] += np.matmul(self.K[t], (self.x[:,t] - self.x_nom[:,t])) + self.k[t]
            self.x[:,t+1] = self.env.integrate_dynamics(self.x[:,t], self.u[:,t])

            epi_cost += self.compute_running_cost(t) 
        
        epi_cost += self.compute_terminal_cost(t+1)
        self.iter_cost_arr.append(float(epi_cost))
        self.iter_traj_arr.append(self.x)
        self.x_nom = self.x.copy()
        

    def backward_pass(self):
        for t in range(self.n - 1, -1, -1):
            if t == self.n - 1:
                V_x, V_xx = self.compute_terminal_cost_derivates(self.x_nom[:,t+1])

            l_x, l_u, l_ux, l_xx, l_uu = self.compute_running_cost_derivatives(self.x_nom[:,t], self.u[:,t], t)
            f_x = np.matrix(self.env.dynamics_x(self.x_nom[:,t], self.u[:,t], self.dt))
            f_u = np.matrix(self.env.dynamics_u(self.x_nom[:,t], self.u[:,t], self.dt))

            V_x = np.matrix(V_x)
            V_xx = np.matrix(V_xx)

            Q_x = l_x + f_x.T*V_x
            Q_u = l_u + f_u.T*V_x
            Q_xx = l_xx + f_x.T*V_xx*f_x
            Q_ux = l_ux + f_u.T*V_xx*f_x
            Q_uu = l_uu + f_u.T*V_xx*f_u

            Q_uu_inv = np.matrix(np.linalg.inv(Q_uu))
            self.K[t] = -Q_uu_inv*Q_ux
            self.k[t] = -Q_uu_inv*Q_u

            V_x = Q_x + self.K[t].T*Q_uu*self.k[t] + self.K[t].T*Q_u + Q_ux.T*self.k[t]
            V_xx = Q_xx + self.K[t].T*Q_uu*self.K[t] + self.K[t].T*Q_ux + Q_ux.T*self.K[t]

    def optimize(self, no_iterations = 10):
        '''
        This function runs ilqr and returs optimal trajectory
        '''
        for n in range(no_iterations):
            self.forward_pass()
            self.backward_pass()
            print("finished iteration {} and the cost is {}".format(n, self.iter_cost_arr[-1]))

            # if n > 3:
            #     if self.iter_cost_arr[-1] > self.iter_cost_arr[-2]:
            #         break

            # plt.plot(self.iter_traj_arr[n][0])
        # plt.plot(self.iter_cost_arr)
        self.forward_pass()
        # plt.plot(self.iter_traj_arr[n+1][0])
        plt.plot(self.u[0])
        plt.show()

        return self.x_nom, self.K, self.u

    def plot(self):
        
        plt.plot((180/np.pi)*self.x[0], label = "new_traj")
        plt.grid()
        plt.legend()
        plt.show()
        