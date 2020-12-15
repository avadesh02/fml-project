# This is an implementation of the double integrator env
# Author : Avadesh Meduri
# Date : 25/06/2020

import numpy as np
from matplotlib import pyplot as plt

# these packages for animating the robot env
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

class DoubleIntegrator:
    
    def __init__(self, mass, length):
        
        self.dt = 0.001
        self.m = mass
        self.length = length # length of the edge of the box (for animation)
        
        self.no_states = 2
        self.no_actions = 1

    def dynamics(self, x, xd, f):
        '''
        This function computes the dynamics (dy/dt = f(y,t)) of the manipulator given
        The state of the system is (position, velocity)
        Input:
            x : position of the block in x axis
            xd : velocity of the block in x axis
            f : force applied on the block in the x axis
        '''
        return xd, f/self.m
    
    def integrate_dynamics_euler(self, x_t, xd_t, f_t):
        '''
        This function integrates the dynamics of the manipulator using euler integration
        for one time step (0.001 sec)
        Input:
            x_t : position of the block in x axis
            xd_t : velocity of the block in x axis
            f_t : force applied on the block in the x axis
        '''
        
        velocity, acceleration = self.dynamics(x_t, xd_t, f_t)
        
        x_t_1 = x_t + velocity*self.dt
        xd_t_1 = xd_t + acceleration*self.dt/self.m
        
        return x_t_1, xd_t_1
    
    def integrate_dynamics(self, states, actions):
        '''
        This function integrates dynamics for one step using the standard api for ilqr
        Input:
            states : the state matrix
            actions : torques
        '''
        return np.array([self.integrate_dynamics_euler(states[0], states[1], actions)], dtype=object)
    
    def dynamics_x(self, state, torque, dt):
        '''
        Returns the derivative of the dynamics with respect to states
        Input:
            state : [joint position  joint velocity]
            torque : torque applied at the end of manipulator
        '''
        A_lin = np.zeros((2,2))
        A_lin[0,1] = 1
        
        A_lin = np.identity(2) + dt*A_lin

        return A_lin

    def dynamics_u(self, state, torque, dt):
        ''' 
        Returns the derivative of the dynamics with respect to torques
        Input:
            state : [joint position  joint velocity]
            torque : torque applied at the end of manipulator
        '''
        B_lin = np.zeros((2,1))
        B_lin[1] = 1/self.m 
        A_lin = np.matrix([[0,1],[0,0]])
        B_lin = B_lin*dt + 0.5*dt**2*A_lin*(B_lin)

        return B_lin

    def reset_double_integrator(self, init_x, init_xd):
        '''
        This function resets the block to a starting state (position, velocity)
        Input:
            init_x : initial position in x
            init_xd : initial velocity in x
        '''
        
        # creating an array sim_data (simulated data) that stores the position, velocity and 
        # force applied on the block throughout the simulation 
        self.sim_data = np.array([[init_x], [init_xd], [0.0]])
        self.t = 0 # time step counter in mili seconds
        
    def reset(self, new_state):
        self.reset_double_integrator(new_state[0], new_state[1])

    def reset_state(self, new_state):
        '''
        This function resets the manipulator to a new position
        Input:
            new_theta : new joint position
            new_theta_dot : new joint velocity
        '''
        sim_data_t_1 = np.array([[new_state[0]], [new_state[1]], [0.0]])
        self.sim_data = np.concatenate((self.sim_data, sim_data_t_1), axis = 1)
        self.t += 1
        
    def step_double_integrator(self, f_t):
        '''
        This function simulates the system using the input force
        '''
        
        self.sim_data[:,self.t][2] = f_t
        
        x_t = self.sim_data[:,self.t][0]
        xd_t = self.sim_data[:,self.t][1]
        
        # integrating dynamics
        x_t_1, xd_t_1 = self.integrate_dynamics_euler(x_t, xd_t, f_t)
        
        # adding the data to sim_data
        self.sim_data = np.concatenate((self.sim_data, [[x_t_1], [xd_t_1], [0.0]]), axis = 1)
        # incrementing time
        self.t += 1
        
    def step(self, f_t, use_euler):
        return self.step_double_integrator(f_t)
        
    def get_position(self):
        '''
        This function returns the location of the block at current time step
        '''
        return self.sim_data[:,self.t][0]
    
    def get_velocity(self):
        '''
        This function returns the velocity of the block at current time step
        '''
        return self.sim_data[:,self.t][1]
    
    def get_state(self):
        '''
        This function returns the state of the block at current time step
        '''
        return self.sim_data[:,self.t][0:2]
    
    def animate(self, freq = 25):
        
        sim_data = self.sim_data[:,::freq]

        fig = plt.figure()
        ax = plt.axes(xlim=(-self.length - 5, self.length+ 5), ylim=(-self.length - 5, self.length + 5))
        text_str = "Double Integrator Animation"
        
        left, = ax.plot([], [], lw=4)
        right, = ax.plot([], [], lw=4)
        top, = ax.plot([], [], lw=4)
        bottom, = ax.plot([], [], lw=4)
        com, = ax.plot([], [], 'o', color='black')
        
        def init():
            left.set_data([], [])
            right.set_data([], [])
            top.set_data([], [])
            bottom.set_data([], [])
            com.set_data([], [])
            
            return left, right, top, bottom, com
        
        def animate(i):
            
            x = sim_data[:,i][0]
            
            left.set_data([x - self.length/2.0, x - self.length/2.0], [0, self.length])
            right.set_data([x + self.length/2.0, x + self.length/2.0], [0, self.length])
            top.set_data([x - self.length/2.0, x + self.length/2.0], [self.length, self.length])
            bottom.set_data([x - self.length/2.0, x + self.length/2.0], [0, 0])
            com.set_data([x,self.length/2.0])
            
            return  left, right, top, bottom, com
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
        ax.grid()
        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=np.shape(sim_data)[1], interval=25, blit=True)

        plt.show()

    def plot(self):
        '''
        This function plots the position, velocity and force
        '''
        
        fig, axs = plt.subplots(3,1, figsize = (10, 10))
        axs[0].plot(self.sim_data[0], label = 'position')
        axs[0].grid()
        axs[0].legend()
        axs[0].set_ylabel("meters")

        axs[1].plot(self.sim_data[1], label = 'velocity')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylabel("meters/sec")
    
        axs[2].plot(self.sim_data[2,:-1], label = 'torque')
        axs[2].grid()
        axs[2].legend()
        axs[2].set_ylabel("Newton/(Meter Second)")
    
    
        plt.show() 