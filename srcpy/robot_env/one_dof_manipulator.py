## This is the implementation of a 1 degree of freedom arm (inverted pendulum)
## Author: Avadesh Meduri
## Date : 9/11/2020

import numpy as np
from matplotlib import pyplot as plt

# these packages for animating the robot env
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


class OneDOFManipulator:
    
    def __init__(self, length, mass, dt = 0.001):
        '''
        This function initialises the class OneDOFManipulatorEnv
        Input:
            length : lenght of the arm
            mass : mass of the rod
        '''
        
        self.length = length
        self.dt = dt # discretization step in seconds
        self.g = 9.81 # gravity vector
        self.m = mass
        # Computing the intertia of the rod about an axis
        # fixed at the end (1/3)ml^2
        self.no_states = 2
        self.no_actions = 1
        self.I = (1/3)*self.m*(self.length**2)
        
    def dynamics(self, theta, theta_dot, torque):
        '''
        This function computes the dynamics (dy/dt = f(y,t)) of the manipulator given
        the current state of (Joint POsition, Joing Velocity)
        Input:
            theta : joint position 
            theta_dot : joint velocity
            torque : torque applied at the end of manipulator
        '''
        
        return np.round(theta_dot,3), np.round((torque - self.m*self.g*np.sin(theta))/self.I,3)
    
    def integrate_dynamics_euler(self, theta_t, theta_dot_t, torque_t):
        '''
        This function integrates the dynamics of the manipulator for one time step (0.001 sec)
        Input:
            theta_t : joint position at current time
            theta_dot_t : joint velocity at current time
            torque_t : torque applied at the end of manipulator at current time
        '''
        
        joint_velocity, joint_acceleration = self.dynamics(theta_t, theta_dot_t, torque_t)
        
        # integrating using euler integration scheme
        # refer to this link for more details : https://en.wikipedia.org/wiki/Euler_method
        
        theta_t_1 = theta_t + joint_velocity*self.dt
        theta_dot_t_1 = theta_dot_t + joint_acceleration*self.dt

        # keeping theta between 0 to 360 degrees
        theta_t_1 = np.sign(theta_t_1)*(abs(theta_t_1)%(2*np.pi))

        return np.round(theta_t_1,3), np.round(theta_dot_t_1,3)
    
    def integrate_dynamics_runga_kutta(self, theta_t, theta_dot_t, torque_t):
        '''
        This function integrates the dynamics of the manipulator for one time step (0.001 sec)
        using runga kutta integration scheme
        Input:
            theta_t : joint position at current time
            theta_dot_t : joint velocity at current time
            torque_t : torque applied at the end of manipulator at current time
        '''
        
        # Runga Kutta is more stable integration scheme as compared to euler
        # refer to this link for more details of runga kutta integration scheme : 
        # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        
        k1_thd, k1_thdd  = self.dynamics(theta_t, theta_dot_t, torque_t)
        k2_thd, k2_thdd  = self.dynamics(theta_t + 0.5*self.dt*k1_thd, theta_dot_t + 0.5*self.dt*k1_thdd, torque_t)
        k3_thd, k3_thdd  = self.dynamics(theta_t + 0.5*self.dt*k2_thd, theta_dot_t + 0.5*self.dt*k2_thdd, torque_t)
        k4_thd, k4_thdd  = self.dynamics(theta_t + self.dt*k3_thd, theta_dot_t + self.dt*k3_thdd, torque_t)
        
        theta_t_1 = theta_t + (1/6)*self.dt*(k1_thd + 2*k2_thd + 2*k3_thd + k4_thd)
        theta_dot_t_1 = theta_dot_t + (1/6)*self.dt*(k1_thdd + 2*k2_thdd + 2*k3_thdd + k4_thdd)
        
        # keeping theta between -360 to 360 degrees
        theta_t_1 = np.sign(theta_t_1)*(abs(theta_t_1)%(2*np.pi))

        return np.round(float(theta_t_1),3), np.round(float(theta_dot_t_1),3) 
    
    def integrate_dynamics(self, states, actions):
        '''
        This function integrates dynamics for one step using the standard api for ilqr
        Input:
            states : the state matrix
            actions : torques
        '''
        return np.array([self.integrate_dynamics_runga_kutta(states[0], states[1], actions)], dtype=object)
    
    def dynamics_x(self, state, torque, dt):
        '''
        Returns the derivative of the dynamics with respect to states
        Input:
            state : [joint position  joint velocity]
            torque : torque applied at the end of manipulator
        '''
        A_lin = np.identity(2)
        A_lin[0,1] = dt
        A_lin[1,0] = -np.round(self.m*self.g*dt*np.cos(state[0])/self.I, 2)

        return A_lin

    def dynamics_u(self, state, torque, dt):
        ''' 
        Returns the derivative of the dynamics with respect to torques
        Input:
            state : [joint position  joint velocity]
            torque : torque applied at the end of manipulator
        '''
        B_lin = np.zeros((2,1))
        B_lin[1] = dt/self.I 

        return B_lin

    def reset_manipulator(self, initial_theta, initial_theta_dot):
        '''
        This function resets the manipulator to the initial position
        Input:
            initial_theta : starting joint position
            initial_theta_dot : starting joint velocity
        '''
        # creating an array sim_data (simulate data) that stores the joint positions,
        # joint velocities and torques at each time step. Each column corresponds to a the 
        # time step. Row 1 contains joint position, Row 2 contains joint velocity, Row 3 contains
        # torque provided by user at the given time step.
        
        self.sim_data = np.array([[initial_theta], [initial_theta_dot], [0.0]])
        self.t = 0 # time counter in milli seconds

    def reset_state(self, new_theta, new_theta_dot):
        '''
        This function resets the manipulator to a new position
        Input:
            new_theta : new joint position
            new_theta_dot : new joint velocity
        '''
        sim_data_t_1 = np.array([[new_theta], [new_theta_dot], [0.0]])
        self.sim_data = np.concatenate((self.sim_data, sim_data_t_1), axis = 1)
        self.t += 1
            
    def step_manipulator(self, torque, use_euler = True):
        '''
        This function integrates the manipulator dynamics for one time step
        Input:
            torque : Input torque at the given time step
        '''
        # storing torque provided by user
        self.sim_data[:,self.t][2] = torque
        
        if use_euler:
            
            theta_t = self.sim_data[:,self.t][0]
            theta_dot_t = self.sim_data[:,self.t][1]
            
            theta_t_1, theta_dot_t_1 = self.integrate_dynamics_euler(theta_t, theta_dot_t, torque)
        
        else:
            theta_t = self.sim_data[:,self.t][0]
            theta_dot_t = self.sim_data[:,self.t][1]
            
            theta_t_1, theta_dot_t_1 = self.integrate_dynamics_runga_kutta(theta_t, theta_dot_t, torque)
            
        # transforming new joint positions and velocity into array form
        sim_data_t_1 = np.array([[theta_t_1], [theta_dot_t_1], [0.0]])
        # adding the data to sim_data
        #Have to check if this line slows things down
        self.sim_data = np.concatenate((self.sim_data, sim_data_t_1), axis = 1)
        # incrementing time
        self.t += 1
        
    def step(self, torque, use_euler=True):
        self.step_manipulator(torque, use_euler)
        
    def get_joint_position(self):
        '''
        This function returns the current joint position (degrees) of the mainpulator
        '''
        return self.sim_data[:,self.t][0]
    
    def get_joint_velocity(self):
        '''
        This function returns the current joint velocity (degrees/sec) of the mainpulator
        '''
        return self.sim_data[:,self.t][1]
    
    def get_joint_state(self):
        return self.sim_data[:,self.t][0:2]
    
    def get_state(self):
        return self.get_joint_state()
    
    def animate(self, freq = 100):
        
        sim_data = self.sim_data[:,::freq]

        fig = plt.figure()
        ax = plt.axes(xlim=(-self.length -1, self.length + 1), ylim=(-self.length -1, self.length + 1))
        text_str = "One Dof Manipulator Animation"
        arm, = ax.plot([], [], lw=4)
        base, = ax.plot([], [], 'o', color='black')
        hand, = ax.plot([], [], 'o', color='pink')
        
        def init():
            arm.set_data([], [])
            base.set_data([], [])
            hand.set_data([], [])
            
            return arm, base, hand
        
        def animate(i):
            theta_t = sim_data[:,i][0]
            
            x = self.length*np.sin(theta_t)
            y = -self.length*np.cos(theta_t)
            
            arm.set_data([0,x], [0,y])
            base.set_data([0, 0])
            hand.set_data([x, y])

            return arm, base, hand
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
        ax.grid()
        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=np.shape(sim_data)[1], interval=25, blit=True)

        plt.show()

    def plot(self):
        '''
        This function plots the joint positions, velocities and torques
        '''
        
        fig, axs = plt.subplots(3,1, figsize = (10, 10))
        axs[0].plot((180/np.pi)*self.sim_data[0], label = 'joint position')
        axs[0].grid()
        axs[0].legend()
        axs[0].set_ylabel("degrees")

        axs[1].plot((180/np.pi)*self.sim_data[1], label = 'joint velocity')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylabel("degrees/sec")
    
        axs[2].plot(self.sim_data[2,:-1], label = 'torque')
        axs[2].grid()
        axs[2].legend()
        axs[2].set_ylabel("Newton/(Meter Second)")
    
    
        plt.show()