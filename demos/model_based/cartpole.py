## This is a demo that generates a plan for the cartpole env using ilqr
## Author : Avadesh Meduri
## Date : 25/11/2020

import os.path
import sys
from matplotlib import pyplot as plt 
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

import numpy as np
from robot_env.cartpole import Cartpole
from model_based.running_cost_functions import QuadraticTrackingCost, ControlRegularizerCost
from model_based.terminal_cost_functions import TerminalQuadraticTrackingCost
from model_based.ilqr import ILQR


# initialising the env
env = Cartpole(5, 1, 5)

init_x = 0
init_theta = 30*(np.pi/180)
init_xd = 0
init_theta_d = 0

env.reset_cartpole(init_x, init_theta, init_xd, init_theta_d)

x_init = env.get_states()
# defining the cost
x_terminal = 0
theta_terminal = 180*(np.pi/180)
xd_terminal = 0
theta_d_terminal = 0

state_terminal = np.array([x_terminal, theta_terminal, xd_terminal, theta_d_terminal])

Q_t = 1e-5*np.identity(4)
Q_t[3,3], Q_t[2,2], Q_t[1,1] = 3.0, 2.0, 1.5e+2
Q_f = 1e-5*np.identity(4)
Q_f[1,1], Q_f[3,3] = 1.5e+4, 2e+2

R_t = 1e-3

ptc = QuadraticTrackingCost(env, state_terminal, Q_t)
tpc = TerminalQuadraticTrackingCost(env, state_terminal, Q_f)
crc = ControlRegularizerCost(env, R_t)
# initialising ilqr
dt = 0.01
T = 2.
no_iterations = 9
ilqr = ILQR(env, dt)
ilqr.initialize(T, x_init)
env.dt = dt
# adding cost
ilqr.add_running_cost(ptc)
ilqr.add_running_cost(crc)
ilqr.add_terminal_cost(tpc)

x_des, K_arr, k_arr = ilqr.optimize(no_iterations)
ilqr.plot()

# # simulating controller
env.dt = 0.001
horizon = int(np.round(T/env.dt, 1)) # duration of simulation steps
r = horizon/ int(np.round(T/dt, 1))

for t in range(horizon):
    state = env.get_states()
    torque = np.matmul(K_arr[int(t//r)],(state - x_des[:,int(t//r)]).transpose()) + k_arr[:,int(t//r)]
    env.step_cartpole(float(torque))

env.animate(15)
env.plot()