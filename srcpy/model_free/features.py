## These are a few classes for features used by actor and critic
## Author : Ilyeech Kishore Rapelli
## Date : 18/11/2020

import numpy as np
class LinearFeatures:

    def __init__(self):
        pass

    def get_sa_features(self, state, action):
        #x = np.array([env.get_joint_position(), env.get_joint_velocity()])
        return [abs(state[0] - 90*(np.pi/180)), state[0] - 90*(np.pi/180), abs(state[1]), state[1], abs(state[0] * action),
                state[0] * action, abs(state[1] * action), state[1] * action]
        
    def get_s_features(self, state):
        #x = np.array([env.get_joint_position(), env.get_joint_velocity()])
        return [abs(state[0] - 90*(np.pi/180)), state[0] - 90*(np.pi/180), abs(state[1]), state[1]]