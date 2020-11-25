## This is a test file that shows how to use the one dof env
## Author : Avadesh Meduri
## Date : 9/11/2020

import os.path
import sys
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

import numpy as np
from robot_env.one_dof_manipulator import OneDOFManipulator


env = OneDOFManipulator(1, 1)

theta_init = 60*(np.pi/180)
env.reset_manipulator(theta_init,0)

horizon = 10000 # duration of simulation steps

for t in range(horizon):
    jp = env.get_joint_position()
    jp_d = env.get_joint_velocity()
    torque = 10*((np.pi/180)*180 - jp) - 1*(jp_d)
    env.step_manipulator(torque, True)

env.animate()
env.plot()