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
from model_based.cost_functions import *
from model_based.ilqr import ILQR


# initialising the env
env = OneDOFManipulator(1, 1)
theta_init = 10*(np.pi/180)
env.reset_manipulator(theta_init,0)

x_init = np.array([env.get_joint_position(), env.get_joint_velocity()])
# defining the cost
x_terminal = np.array([np.pi, 0]) 
Q_t = 100*np.identity(2)
Q_f = 1e+1*np.identity(2)
R_t = 1e-5

ptc = PositionTrackingCost(env, x_terminal, Q_t)
tpc = TerminalPositionTrackingCost(env, x_terminal, Q_f)
crc = ControlRegularizerCost(env, R_t)

# initialising ilqr
dt = 0.001
T = 0.5
no_iterations = 50
ilqr = ILQR(env, dt)
ilqr.initialize(T, ptc, crc, tpc, x_init)
x_des, K_arr, k_arr = ilqr.optimize(no_iterations)
ilqr.plot()
# simulating controller

horizon = int(np.round(T/dt, 1)) # duration of simulation steps


for t in range(horizon):
    jp = env.get_joint_position()
    jp_d = env.get_joint_velocity()
    state = np.array([jp, jp_d], dtype=object)
    torque = np.matmul(K_arr[:,:,t],(state - x_des[:,t]).transpose()) + k_arr[:,t]
    env.step_manipulator(float(torque), True)

env.animate(10)
# env.plot()