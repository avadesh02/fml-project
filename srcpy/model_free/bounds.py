## This file contains the implementation of cost functions
## Author : Ilyeech Kishore Rapelli
## Date : 15/12/2020

from math import radians

class DIBound:

    def __init__(self, env, state_terminal):
        '''
        Input:
            env : dynamics of the system
        '''
        self.env = env
        assert len(state_terminal) == env.no_states
        self.state_terminal = state_terminal

    def end_episode(self, state):
        '''
        Input:
            state : state at time t
        '''
        if(abs(state[0] - self.state_terminal[0]) > 4):
            return True
        elif(abs(state[1] - self.state_terminal[1]) > 1000):
            return True
        else:
            return False
    
class CartPoleBound:

    def __init__(self, env, state_terminal):
        '''
        Input:
            env : dynamics of the system
        '''
        self.env = env
        assert len(state_terminal) == env.no_states
        self.state_terminal = state_terminal

    def end_episode(self, state):
        '''
        Input:
            state : state at time t
        '''
        if(abs(state[0] - self.state_terminal[0]) > 1.5):
            return True
        elif(abs(state[1] - self.state_terminal[1]) > radians(60)):
            return True
        elif(abs(state[2] - self.state_terminal[2]) > 1000):
            return True
        elif(abs(state[3] - self.state_terminal[3]) > 1000):
            return True
        else:
            return False
            #if(abs(state[0]) > 1.5 or abs(state[1]) > radians(60)):#Section 4.1 of NAC paper
            #    #if(self.DEBUG):
            #    print("Out of bounds. Ending episode " + str(episode) + " in step (index)" + str(step))
            #    break