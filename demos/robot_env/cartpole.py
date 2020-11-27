## This is a test file that shows how to use the cartpole
## Author : Avadesh Meduri
## Date : 25/11/2020

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
from robot_env.cartpole import Cartpole

env = Cartpole(5, 1, 5)

init_x = 0
init_theta = 250*(np.pi/180)
init_xd = 0
init_theta_d = 0

env.reset_cartpole(init_x, init_theta, init_xd, init_theta_d)

horizon = 10000 # duration of simulation steps

for t in range(horizon):
    
    state = env.get_states()
    # torque = -20*(state[0] - 5*np.sin(np.pi*0.001*t))
    torque = 0
    env.step_cartpole(torque)

env.animate()
env.plot()
