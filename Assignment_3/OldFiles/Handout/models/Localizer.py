
#
# The Localizer binds the models together and controls the update cycle in its "update" method.
#

from os import read
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import random

from models import StateModel,TransitionModel,ObservationModel,RobotSimAndFilter

class Localizer:
    def __init__(self, sm:StateModel):

        self.__sm = sm

        self.__tm = TransitionModel(self.__sm)
        self.__om = ObservationModel(self.__sm)

        # change in initialise in case you want to start out with something else
        # initialise can also be called again, if the filtering is to be reinitialised without a change in size
        self.initialise()

    # retrieve the transition model that we are currently working with
    def get_transition_model(self) -> np.array:
        return self.__tm

    # retrieve the observation model that we are currently working with
    def get_observation_model(self) -> np.array:
        return self.__om

    # the current true pose (x, h, h) that should be kept in the local variable __trueState
    def get_current_true_pose(self) -> (int, int, int):
        x, y, h = self.__sm.state_to_pose(self.__trueState)
        return x, y, h

    # the current probability distribution over all states
    def get_current_f_vector(self) -> np.array(float):
        return self.__probs

    # the current sensor reading (as position in the grid). "Nothing" is expressed as None
    def get_current_reading(self) -> (int, int):
        ret = None
        if self.__sense != None:
            ret = self.__sm.reading_to_position(self.__sense)
        return ret;

    # get the currently most likely position, based on single most probable pose
    def most_likely_position(self) -> (int, int):
        return self.__estimate

    ################################### Here you need to really fill in stuff! ##################################
    # if you want to start with something else, change the initialisation here!
    #
    # (re-)initialise for a new run without change of size
    def initialise(self):
        self.__trueState = random.randint(0, self.__sm.get_num_of_states() - 1)
        self.__sense = None
        self.__probs = np.ones(self.__sm.get_num_of_states()) / (self.__sm.get_num_of_states())
        self.__estimate = self.__sm.state_to_position(np.argmax(self.__probs))  
        
        self.__rs = RobotSimAndFilter.RobotSim(self.__sm,self.__trueState)
        self.__HMM = RobotSimAndFilter.HMMFilter(self.__sm,self.__tm,self.__om)
   
    def update(self) -> (bool, int, int, int, int, int, int, int, int, np.array(1)) :
        
        self.__rs.move()
        self.__trueState = self.__rs.getState()
        x_cord, y_cord, heading = self.__sm.state_to_pose(self.__trueState)
        print("Robot location x :{} y: {} h: {}  ".format(x_cord,y_cord,heading))

        self.__sense  = self.__rs.senseLoc()
        self.__probs,self.__estimate = self.__HMM.update(self.__sense ,self.__probs)

        # this block can be kept as is --------------------------------
        ret = False  # in case the sensor reading is "nothing" this is kept...
        tsX, tsY, tsH = self.__sm.state_to_pose(self.__trueState)
        srX = -1
        srY = -1
        if self.__sense != None:
            srX, srY = self.__sense
            ret = True
            
        eX, eY = self.__estimate
        # -------------------------------------------------------------
        print("Sensed state: {},{}".format(eX,eY))
        error = abs(tsX-eX)+(abs(tsY-eY))               
        
        # if you use the visualisation (dashboard), this return statement needs to be kept the same
        # or the visualisation needs to be adapted (your own risk!)
        return ret, tsX, tsY, tsH, srX, srY, eX, eY, error, self.__probs
