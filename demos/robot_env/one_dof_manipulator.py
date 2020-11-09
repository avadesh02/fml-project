## This is a test file that shows how to use the one dof env
## Author : Avadesh Meduri
## Date : 9/11/2020

import os.path
import sys
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

from robot_env.one_dof_manipulator import OneDOFManipulator


env = OneDOFManipulator(1, 1)
env.reset_manipulator(60,0)

horizon = 20000 # duration of simulation steps

for t in range(horizon):
    torque = 0
    env.step_manipulator(torque, True)

env.animate()
env.plot()