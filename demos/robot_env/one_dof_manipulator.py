## This is a test file that shows how to use the one dof env
## Author : Avadesh Meduri
## Date : 9/11/2020

import os.path
import sys
print("Printing some details of the Compiler:\nPython version")
print (sys.version)
print("Version info.")
print (sys.version_info)

curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

import numpy as np
from robot_env.one_dof_manipulator import OneDOFManipulator


env = OneDOFManipulator(1, 2)
theta_init = -60*(np.pi/180)
env.reset_manipulator(theta_init,0)

horizon = 50000 # duration of simulation steps


for t in range(horizon):
    jp = env.get_joint_position()
    jp_d = env.get_joint_velocity()
    torque = 10*np.abs((np.pi/180)*180 - jp) - 2*(jp_d)
    env.step_manipulator(torque, True)

env.animate()
env.plot()