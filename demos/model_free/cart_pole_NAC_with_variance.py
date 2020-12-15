## This is a demo for the One-step Actor Critic (from the book)
## one episode only on 20th Nov
## Author : Ilyeech Kishore Rapelli
## Date : 20/11/2020

import os.path
import sys
from matplotlib import pyplot as plt 
from math import radians
#from numpy.lib.tests.test_format import dt1
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../srcpy/'))
sys.path.append(cdir)

import numpy as np
from robot_env.cartpole import Cartpole
from model_free.cost_functions import *
from model_free.NAC_actor_with_variance_parameter import *
from model_free.NAC_critic import *
from model_free.features import *


DEBUG = 0
env = Cartpole(5, 5, 5)
#env = Cartpole(1.0, 0.15, 0.75)#From section 4.1 of the NAC paper

init_x = 0
init_theta = radians(160)
init_xd = 0
init_theta_d = 0

env.reset_cartpole(init_x, init_theta, init_xd, init_theta_d)

state_init = np.array([init_x, init_theta, init_xd, init_theta_d])

#Initializing the RL model
# defining the cost
x_terminal = 0
theta_terminal = radians(180)
xd_terminal = 0
theta_d_terminal = 0

state_terminal = np.array([x_terminal, theta_terminal, xd_terminal, theta_d_terminal])
Q_t = 1e-5*np.identity(4)
Q_t[0,0], Q_t[1,1], Q_t[2,2], Q_t[3,3] = [1.25, 1., 12., 0.25]#Section 4.1 pf NAC paper
Q_f = Q_t
R_t = 0.01#Section 4.1 pf NAC paper
dt = 1/60#Section 4.1 pf NAC paper

qc = QuadraticCost(env, state_terminal, Q_t)#as of now, keeping this time-invariant Q_t
terminal_qc = QuadraticTerminalCost(env, state_terminal, Q_f)
crc = ControlRegularizerCost(env, R_t)
costs_combined = Cost()
costs_combined.initialize(terminal_qc, qc, crc)
print(qc.compute(np.array([3, 0.1, 0, 0]), 0).item())
print(qc.compute(np.array([12, 10.1, 0, 5.1]), 0).item())
print(crc.compute(3, 0).item())
print(crc.compute(113, 0).item())
#defining the features
lf = LinearFeaturesWithOneCartpole()
#defining the critic
#dt = 0.1
env.dt = dt
alpha_critic = 0.0001
gamma = 1.0#1.0#0.999 was stabler#Jan Peters says gamma < 1 destroys learning performance
lfc = LinearFeaturesNACCritic(env, dt, DEBUG)
print(state_init)
print(lf.get_s_features(state_init))
print(lf.get_s_features(np.array([1, 2.3, 0.5, 20])))
feature_size = len(lf.get_s_features(state_init))
print(feature_size)
critic_init = np.random.normal(0.,0.1,feature_size)#np.array([0.0, 0.0, 0.0, 0.0])
print("Initial Critic: " + str(critic_init))
lfc.initialize(alpha_critic, gamma, critic_init, lf)
#defining the actor
alpha_actor = 0.01
lfga = LinearFeaturesGaussianNACActor(env, dt, DEBUG)
actor_init = np.random.normal(0.,100,feature_size + 1)#np.array([0.0, 0.0, 0.0, 0.0])
actor_init[feature_size - 1] = np.random.normal(1., 0.1, 1)#eta can't be big in the beginning
# +1 For the sake of eta parameter of section 4.1 of the paper
#actor_init = np.array([5.5, 10.0, -80.0, -20.0, 10.0, 1.0])#With eta = 1000, SVD didn't converge
#actor_init = np.array([5.71, 11.3, -82.1, -21.6, 100.0, 1.0])
print("Initial Actor: " + str(actor_init))
T = 100 * dt
lfga.initialize(T, alpha_actor, gamma, lfc, actor_init, lf, costs_combined, state_init)


#Optimizing the RL model
#Should convert to episodic
use_euler = False#False means Runge-Kutta
no_iterations = 10000
no_episodes = 20
max_episode_length = int(np.round(T/dt, 1))
print(dt, T, no_iterations, no_episodes, max_episode_length)
if(no_episodes < feature_size):
    print("\nERROR: Natural Actor Critic requires no of episodes at least equal to" + 
          "\n or greater than 1 + 1 + no of policy_parameters. If w doesn't" +
          "\n converge in the very first attempt, it requires more.")
    sys.exit()
lfga.optimize(no_iterations, no_episodes, max_episode_length, use_euler = use_euler)

print("\nSimulation based on the learned model:")
# simulating controller
simulation_T = 10 * dt
horizon = int(np.round(simulation_T/dt, 1)) # duration of simulation steps
for t in range(horizon):
    jp, jp_d,theta, theta_d = env.get_state()
    state = np.array([jp, jp_d,theta, theta_d], dtype=object)
    torque, grad = lfga.sample_action(state)
    #env.step_manipulator(float(torque), use_euler = use_euler)
    env.step(float(torque))


print("\n\nInitial Actor: " + str(actor_init))
print("Final Actor: " + str((lfga.parameters).tolist()))
print("Actor norm ratio: " + str(np.linalg.norm(lfga.parameters[0:lfga.parameters_size - 1]) / np.linalg.norm(np.array(actor_init[0:lfga.parameters_size - 1]))))

print("T=", max_episode_length, "*dt, dt=", dt, ", iter=", no_iterations, 
      ", epis=", no_episodes, ",gamma=", gamma, ",alpha=", alpha_actor, ",start=,sigma=", 
      "- CP - end ANGLE")

#lfga.plot()#?
#lfga.plot_vel()
#lfga.plot_torque()
lfga.plot_end_angle()
lfga.plot_end_position()
#lfga.plot_episode_cost(0.997)

env.animate(1)
#env.plot()

#lfc.plot_policy()
#lfga.plot_policy()