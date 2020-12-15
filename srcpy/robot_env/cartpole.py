## This is an implementation of the cartpole environment
## Author : Avadesh Meduri
## Date : 25/11/2020

import numpy as np
from matplotlib import pyplot as plt

# these packages for animating the robot env
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

class Cartpole:

    def __init__(self, mass_cart, mass_pole, length_pole):

        self.dt = 0.001
        self.g = 9.81
        self.mc = mass_cart
        self.mp = mass_pole
        self.lp = length_pole
        self.length = 1

        self.no_states = 4
        self.no_actions = 1

    def dynamics(self, x, theta, xd, theta_d, actions):
        '''
        This function computes the dynamics of the system (dy/dt = f(y,t)) 
        Input:
            x : position of the cart pole 
            theta : joint position of the pole
            xd : velocity of the cartpole
            theta_d : joint velocity of the cartpole
            action : force applied to the base
        '''

        dy = np.zeros(4)
        dy[0] = xd
        dy[1] = theta_d
        dy[2] = (1/(self.mc + self.mp*np.sin(theta)**2))*(actions + self.mp*np.sin(theta)*(self.lp*theta_d**2 + self.g*np.cos(theta)))
        dy[3] = (1/(self.lp*(self.mc + self.mp*np.sin(theta)**2)))*(-actions*np.cos(theta) - self.mp*self.lp*(theta_d**2)*0.5*np.sin(2*theta) - (self.mc + self.mp)*self.g*np.sin(theta))

        # print(self.mc + self.mp*np.sin(theta)**2, theta)
        return dy

    def dynamics_state_derivative(self, x, theta, xd, theta_d, actions):
        '''
        computes derivative of the dynamics wrt to states
        Input:
            x : position of the cart pole 
            theta : joint position of the pole
            xd : velocity of the cartpole
            theta_d : joint velocity of the cartpole
            action : force applied to the base
        '''
        # https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-832-underactuated-robotics-spring-2009/readings/MIT6_832s09_read_ch03.pdf
        A_x = np.zeros((4,4))
        A_x[0,2] = 1.0
        A_x[1,3] = 1.0

        den = (self.mc + self.mp*np.sin(theta)**2)    
        A_x[2,1] = (-self.mp*np.sin(2*theta)*(actions + self.mp*np.sin(theta)*(self.lp*theta_d**2 + self.g*np.cos(theta))))/den**2 \
                    + (self.lp*self.mp*np.cos(theta)*theta_d**2 + self.mp*self.g*np.cos(2*theta))/den
        
        A_x[2,3] = 2*self.mp*self.lp*theta_d*np.sin(theta)/den

        A_x[3,1] = (self.mp*np.sin(2*theta)*(actions*np.cos(theta) + 0.5*self.mp*self.lp*(theta_d**2)*np.sin(2*theta) + (self.mc + self.mp)*self.g*np.sin(theta)))/(self.lp*(den**2)) + \
                    (actions*np.sin(theta) - self.mp*self.lp*np.cos(2*theta)*theta_d**2 - (self.mc + self.mp)*self.g*np.cos(theta))/(self.lp*den)
        A_x[3,3] = (-self.mp*self.lp*theta_d*(np.sin(2*theta)))/(self.lp*den)

        return A_x


    def integrate_dynamics(self, state, actions):
        '''
        This function integrates dynamics for one step using the standard api for ilqr
        Input:
            states : the state matrix
            actions : torques
        '''
    
        return np.add(state, self.dynamics(state[0], state[1], state[2], state[3], actions)*self.dt)

    def dynamics_x(self, state, actions, dt):
        '''
        ## fix the dt requirement here since it is already defined in the init of ilqr
        '''
        A_lin = np.identity(4)
        A_lin += self.dynamics_state_derivative(state[0], state[1], state[2], state[3], actions)*dt

        return A_lin

    def dynamics_u(self, state, actions, dt):
        '''
        computes the descrete dynamics of the system wrt to actions
        Input:
            states : the state matrix
            actions : torques
        '''
        B_lin = np.zeros((4,1))
        den = (self.mc + self.mp*np.sin(state[1])**2)
        B_lin[2] = (dt/den)
        B_lin[3] = (dt/(self.lp*den))*(-np.cos(state[1]))

        return B_lin

    def reset_cartpole(self, init_x, init_theta, init_xd, init_theta_d):
        '''
        This function resets the system to initial position
        Input:
            initial states
        '''
        self.sim_data = np.array([[init_x], [init_theta], [init_xd], [init_theta_d], [0.0]])
        self.t = 0 # time step counter in mili seconds

    def reset(self, new_state):
        self.reset_cartpole(new_state[0], new_state[1], new_state[2], new_state[3])

    def reset_state(self, new_state):
        '''
        This function resets the system to initial position without resetting history
        Input:
            new initial state
        '''
        sim_data_t_1 = np.array([[new_state[0]], [new_state[1]], [new_state[0]], [new_state[1]], [0.0]])
        self.sim_data = np.concatenate((self.sim_data, sim_data_t_1), axis = 1)
        self.t += 1

    def step_cartpole(self, actions):
        '''
        This function simlates the system
        Input:
            actions : force at the curret time step
        '''

        self.sim_data[:,self.t][4] = actions
        ## adding state at next time step
        self.sim_data = np.concatenate((self.sim_data, np.zeros((5,1))), axis = 1)

        self.sim_data[:,self.t+1][0:4] = self.integrate_dynamics(self.sim_data[:,self.t][0:4], self.sim_data[:,self.t][4])
        # keeping theta between -360 to 360 degrees
        self.sim_data[1] = np.sign(self.sim_data[1])*(abs(self.sim_data[1])%(2*np.pi))
        
        self.t += 1
        
    def step(self, actions, use_euler):
        self.step_cartpole(actions)

    def get_states(self):
        '''
        This function returns the state of the system at the current time step
        '''
        return self.sim_data[:,self.t][0:4]
    
    def get_state(self):
        '''
        This function returns the state of the block at current time step
        '''
        return self.get_states()

    def animate(self, freq = 25):
        
        sim_data = self.sim_data[:,::freq]

        fig = plt.figure()
        ax = plt.axes(xlim=(-self.length - 10, self.length+ 10), ylim=(-self.length - 10, self.length + 10))
        text_str = "Cartpole Animation"
        
        left, = ax.plot([], [], lw=4)
        right, = ax.plot([], [], lw=4)
        top, = ax.plot([], [], lw=4)
        bottom, = ax.plot([], [], lw=4)
        pole, = ax.plot([], [], lw=4)
        com, = ax.plot([], [], 'o', color='black')
        

        def init():
            left.set_data([], [])
            right.set_data([], [])
            top.set_data([], [])
            bottom.set_data([], [])
            com.set_data([], [])
            pole.set_data([], [])
            

            return left, right, top, bottom, com, pole
        
        def animate(i):
            
            x = sim_data[:,i][0]
            theta = sim_data[:,i][1]

            left.set_data([x - self.length/2.0, x - self.length/2.0], [0, self.length])
            right.set_data([x + self.length/2.0, x + self.length/2.0], [0, self.length])
            top.set_data([x - self.length/2.0, x + self.length/2.0], [self.length, self.length])
            bottom.set_data([x - self.length/2.0, x + self.length/2.0], [0, 0])
            com.set_data([x,self.length/2.0])
            
            pole.set_data([x, x + self.lp*np.sin(theta)], [self.length/2.0, self.length/2.0 - self.lp*np.cos(theta)])

            return  left, right, top, bottom, com, pole
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=15,
        verticalalignment='top', bbox=props)
        ax.grid()
        anim = FuncAnimation(fig, animate, init_func=init,
                                       frames=np.shape(sim_data)[1], interval=25, blit=True)

        plt.show()
        anim.save("cartpole_env.mp4")

    def plot(self):
        '''
        This function plots the joint positions, velocities and torques
        '''
        
        fig, axs = plt.subplots(5,1, figsize = (10, 10))
        axs[0].plot(self.sim_data[0], label = 'cart position')
        axs[0].grid()
        axs[0].legend()
        axs[0].set_ylabel("meters")

        axs[1].plot((180/np.pi)*self.sim_data[1], label = 'joint position')
        axs[1].grid()
        axs[1].legend()
        axs[1].set_ylabel("degrees")
    
        axs[2].plot(self.sim_data[2], label = 'cart velocity')
        axs[2].grid()
        axs[2].legend()
        axs[2].set_ylabel("meters/sec")

        axs[3].plot((180/np.pi)*self.sim_data[3], label = 'joint velocity')
        axs[3].grid()
        axs[3].legend()
        axs[3].set_ylabel("degrees/sec")
    
        axs[4].plot(self.sim_data[4,:-1], label = 'force')
        axs[4].grid()
        axs[4].legend()
        axs[4].set_ylabel("Newton")
    
    
        plt.show()
