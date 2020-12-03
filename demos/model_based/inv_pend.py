## This is a demo for the ilqr
## Author : Avadesh Meduri
## Date : 11/11/2020

import os.path
import sys
from matplotlib import pyplot as plt 
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

import numpy as np
from robot_env.one_dof_manipulator import OneDOFManipulator
from model_based.running_cost_functions import QuadraticTrackingCost, ControlRegularizerCost
from model_based.terminal_cost_functions import TerminalQuadraticTrackingCost
from model_based.ilqr import ILQR


# initialising the env
env = OneDOFManipulator(1, 1)
theta_init = 0*(np.pi/180)
env.reset_manipulator(theta_init,0)
x_init = np.array([env.get_joint_position(), env.get_joint_velocity()])
# defining the cost
x_terminal = np.array([180*(np.pi/180), 0]) 
Q_t = np.identity(2)
Q_t[0,0], Q_t[1,1] = [100, 20]
Q_f = Q_t
R_t = 1e-2

ptc = QuadraticTrackingCost(env, x_terminal, Q_t)
tpc = TerminalQuadraticTrackingCost(env, x_terminal, Q_f)
crc = ControlRegularizerCost(env, R_t)
# initialising ilqr
dt = 0.01
T = 2
no_iterations = 20
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
env.dt = 0.01
horizon = int(np.round(T/env.dt, 1)) # duration of simulation steps
r = horizon/ int(np.round(T/dt, 1))

for t in range(horizon):
    jp = env.get_joint_position()
    jp_d = env.get_joint_velocity()
    state = np.array([jp, jp_d], dtype=object)
    torque = np.matmul(K_arr[int(t//r)],(state - x_des[:,int(t//r)]).transpose()) + k_arr[:,int(t//r)]
    env.step_manipulator(float(torque), True)

env.animate(1)
# env.plot()
