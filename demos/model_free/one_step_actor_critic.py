## This is a demo for the One-step Actor Critic (from the book)
## one episode only on 20th Nov
## Author : Ilyeech Kishore Rapelli
## Date : 20/11/2020

import os.path
import sys
from matplotlib import pyplot as plt 
#from numpy.lib.tests.test_format import dt1
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

import numpy as np
from robot_env.one_dof_manipulator import OneDOFManipulator
from model_free.cost_functions import *
from model_free.actor import *
from model_free.critic import *
from model_free.features import *


# initialising the env
env = OneDOFManipulator(1, 0.1)
theta_init = 194*(np.pi/180)
env.reset_manipulator(theta_init,0)

state_init = np.array([env.get_joint_position(), env.get_joint_velocity()])

#Initializing the RL model
# defining the cost
state_terminal = np.array([np.pi/2, 0]) 
Q_t = 10*np.identity(2)
qc = QuadraticCost(env, state_terminal, Q_t)
#defining the features
lf = LinearFeatures()#may have to pass env?
#defining the critic
dt = 0.001
alpha_critic = 0.001
gamma = 1.0
lfc = LinearFeaturesCritic(env, dt)
critic_init = np.array([0.0, 0.0, 0.0, 0.0])
lfc.initialize(alpha_critic, gamma, critic_init, lf)
#defining the actor
alpha_actor = 0.001
lfga = LinearFeaturesGaussianActor(env, dt)
actor_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
T = 0.5
lfga.initialize(T, alpha_actor, gamma, lfc, actor_init, lf, qc, state_init)


#Optimizing the RL model
no_iterations = 30
lfga.optimize(no_iterations)
lfga.plot()#?
# simulating controller

horizon = int(np.round(T/dt, 1)) # duration of simulation steps
horizon = 100

for t in range(horizon):
    jp = env.get_joint_position()
    jp_d = env.get_joint_velocity()
    state = np.array([jp, jp_d], dtype=object)
    torque, grad = lfga.sample_action(state)
    env.step_manipulator(float(torque), True)

env.animate(1)
# env.plot()