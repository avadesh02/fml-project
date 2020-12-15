## This file contains the implementation of cost functions
## Author : Ilyeech Kishore Rapelli
## Date : 14/12/2020

class FixedPosition:
    def __init__(self,x_nom):
        self.x_nom = x_nom
        
    def reset(self):
        return self.x_nom 