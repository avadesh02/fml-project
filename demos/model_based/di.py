## This is a demo for the ilqr
## Author : Avadesh Meduri
## Date : 12/11/2020

import os.path
import sys
from matplotlib import pyplot as plt 
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

import numpy as np
from robot_env.double_integrator import DoubleIntegrator
from model_based.running_cost_functions import QuadraticTrackingCost, ControlRegularizerCost
from model_based.terminal_cost_functions import TerminalQuadraticTrackingCost
from model_based.ilqr import ILQR


# initialising the env
env = DoubleIntegrator(1, 1)
env.reset_double_integrator(0,0)
x_init = np.array([env.get_position(), env.get_velocity()])
# defining the cost
x_terminal = np.array([2, 0]) 
Q_t = 1e+2*np.identity(2)
Q_f = np.identity(2)
R_t = 0

ptc = QuadraticTrackingCost(env, x_terminal, Q_t)
tpc = TerminalQuadraticTrackingCost(env, x_terminal, Q_f)
crc = ControlRegularizerCost(env, R_t)
# initialising ilqr
dt = 0.1
T = 5.0
env.dt = dt
no_iterations = 1
ilqr = ILQR(env, dt)
ilqr.initialize(T, x_init)
# adding cost
ilqr.add_running_cost(ptc)
ilqr.add_running_cost(crc)
ilqr.add_terminal_cost(tpc)

x_des, K_arr, k_arr = ilqr.optimize(no_iterations)
# ilqr.plot()
# # simulating controller
env.dt = 0.001
horizon = int(np.round(T/env.dt, 1)) # duration of simulation steps
r = int(horizon/ int(np.round(T/dt, 1)))

for t in range(horizon-r):
    x = env.get_position()
    xd = env.get_velocity()
    state = np.array([x, xd], dtype=object)
    torque = np.matmul(K_arr[int(t//r)],(state - x_des[:,int(t//r)]).transpose()) + k_arr[:,int(t//r)]
    env.step_double_integrator(float(torque))

env.animate(10)
env.plot()