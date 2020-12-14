## These are a few classes for features used by actor and critic
## Author : Ilyeech Kishore Rapelli
## Date : 18/11/2020

import numpy as np
import math
import sys

class LinearFeatures:

    def __init__(self):
        pass
        
    def get_s_features(self, state):
        #x = np.array([env.get_joint_position(), env.get_joint_velocity()])
        #return [state[0] - 90*(np.pi/180), abs(state[0] - 90*(np.pi/180)), state[1], abs(state[1]),
        #       (state[0] - 90*(np.pi/180)) * (state[0] - 90*(np.pi/180)), state[1] * state[1]]
        x_sign = math.copysign(1.0, state[0] - math.radians(90))
        x_abs = abs(state[0])
        #x_abs_log = math.log(x_abs)
        #x_abs_log_signed = x_abs_log * x_sign
        y_sign = math.copysign(1.0, state[1])
        y_abs = abs(state[1])
        #print(y_abs)
        #y_abs_log = math.log(max(y_abs, sys.float_info.min))#in case y_abs = 0.0
        #y_abs_log = math.log(y_abs)
        #y_abs_log_signed = y_abs_log * y_sign
        
        #return [x_abs_log_signed, y_abs_log_signed, x_abs_log_signed * y_abs_log_signed]#This XOR like feature is important
        #return [x_abs_log_signed, y_abs_log_signed, x_abs_log_signed * y_abs_log_signed,
        #        state[0] - math.radians(90), state[1]]
        return [state[0] - 2, state[1] - 0]
        #return [y_abs_log_signed, x_abs_log_signed * y_abs_log_signed]
        #return [state[0] - 90*(np.pi/180), state[1], 
        #        (state[0] - 90*(np.pi/180)) * (state[1])* (state[1])* (state[1]),
        #        (state[0] - 90*(np.pi/180)) * (state[0] - 90*(np.pi/180)) * (state[0] - 90*(np.pi/180)) * (state[1]),
        #        (state[0] - 90*(np.pi/180)) * (state[1])]#This XOR like feature is important
        #return [(state[0] - 90*(np.pi/180)) * (state[1])]
     
    def get_sa_features(self, state, action):
        #x = np.array([env.get_joint_position(), env.get_joint_velocity()])
        #return [abs(state[0] - 90*(np.pi/180)), state[0] - 90*(np.pi/180), abs(state[1]), 
        #        state[1], abs(state[0] * action),
        #        state[0] * action, abs(state[1] * action), state[1] * action]
        return [state[0] - 90*(np.pi/180), abs(state[0] - 90*(np.pi/180)), 
                state[1], abs(state[1]), action, abs(action)]
        
class LinearFeaturesWithOne:

    def __init__(self):
        pass
        
    def get_s_features(self, state):
        features = [state[0] - 2, state[1] - 0]
        features.append(1.0)
        return features
        #return features.append(1.0) #returned None
        
class LinearFeaturesWithOneCartpole:

    def __init__(self):
        pass
        
    def get_s_features(self, state):
        #features = [state[0], state[1] - math.radians(180), state[2], state[3]]
        #features = [state[0], math.radians(180) - state[1], -state[2], -state[3]]#Section 4.1 of the NAC paper#-state[2]????
        features = [state[0], math.radians(180) - state[1], state[2], -state[3], state[0]*math.radians(180) - state[1]]#Section 4.1 of the NAC paper]
        features.append(1.0)
        return features
        #return features.append(1.0) #returned None