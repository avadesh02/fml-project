## This is a test file that shows how to use the one dof env
## Author : Avadesh Meduri
## Date : 9/11/2020

import os.path
import sys
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

from robot_env.two_dof_manipulator import TwoDOFManipulator

# simulation of Env
env = TwoDOFManipulator(1, 1, 1, 1)
env.reset_manipulator(45, 0, 0, 0)

horizon = 5000 # duration of simulation steps

for t in range(horizon):
    tau1 = 0
    tau2 = 0
    env.step_manipulator(tau1, tau2)

env.animate()
env.plot()