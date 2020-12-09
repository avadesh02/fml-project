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
from robot_env.double_integrator import DoubleIntegrator
from model_free.cost_functions import *
from model_free.NAC_actor import *
from model_free.NAC_critic import *
from model_free.features import *


DEBUG = 0

# initialising the env
env = DoubleIntegrator(1, 0.1)
x_init = 0
env.reset_double_integrator(x_init,0.001)

state_init = np.array([0.0, 0.0])

#Initializing the RL model
# defining the cost
state_terminal = np.array([2, 0]) #horizontal position
Q_t = 10*np.identity(2)
Q_f = 1e+4*np.identity(2)
R_t = 1e-4
qc = QuadraticCost(env, state_terminal, Q_t)#as of now, keeping this time-invariant Q_t
terminal_qc = QuadraticTerminalCost(env, state_terminal, Q_f)
crc = ControlRegularizerCost(env, R_t)
costs_combined = Cost()
costs_combined.initialize(terminal_qc, qc, crc)
print(qc.compute(np.array([3, 0.1]), 0).item())
print(qc.compute(np.array([12, 10.1]), 0).item())
print(crc.compute(3, 0).item())
print(crc.compute(113, 0).item())
#defining the features
lf = LinearFeaturesWithOne()
#defining the critic
dt = 0.001
alpha_critic = 0.0001
gamma = 1.0#1.0#0.999 was stabler#Jan Peters says gamma < 1 destroys learning performance
lfc = LinearFeaturesNACCritic(env, dt, DEBUG)
print(state_init)
print(lf.get_s_features(state_init))
feature_size = len(lf.get_s_features(state_init))
print(feature_size)
critic_init = np.random.normal(0.,0.1,feature_size)#np.array([0.0, 0.0, 0.0, 0.0])
print("Initial Critic: " + str(critic_init))
lfc.initialize(alpha_critic, gamma, critic_init, lf)
#defining the actor
alpha_actor = 0.0001
lfga = LinearFeaturesGaussianNACActor(env, dt, DEBUG)
actor_init = np.random.normal(0.,0.1,feature_size)#np.array([0.0, 0.0, 0.0, 0.0])
print("Initial Actor: " + str(actor_init))
T = 1000 * dt
lfga.initialize(T, alpha_actor, gamma, lfc, actor_init, lf, costs_combined, state_init)


#Optimizing the RL model
#Should convert to episodic
use_euler = False#False means Runge-Kutta
no_iterations = 100
no_episodes = 10
max_episode_length = int(np.round(T/dt, 1))
if(no_episodes < feature_size):
    print("Natural Actor Critic requires no of episodes at least equal" + 
          " or greater than 1 + 1 + no of policy_parameters. If w doesn't" +
          " converge fast, it requires more.")
    sys.exit()
lfga.optimize(no_iterations, no_episodes, max_episode_length, use_euler = use_euler)
#lfga.plot()#?
#lfga.plot_vel()
#lfga.plot_torque()
#lfga.plot_episode_cost(0.997)
#lfc.plot_policy()
lfga.plot_policy()

print("Simulation based on the learned model:\n")
# simulating controller
simulation_T = 100 * dt
horizon = int(np.round(simulation_T/dt, 1)) # duration of simulation steps
for t in range(horizon):
    #jp = env.get_joint_position()
    #jp_d = env.get_joint_velocity()
    jp, jp_d = env.get_state()
    state = np.array([jp, jp_d], dtype=object)
    torque, grad = lfga.sample_action(state)
    #env.step_manipulator(float(torque), use_euler = use_euler)
    env.step_double_integrator(float(torque))

env.animate(50)
env.plot()