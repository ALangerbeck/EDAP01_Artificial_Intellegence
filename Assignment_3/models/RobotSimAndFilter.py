
from os import remove
import random as rnd
from types import NoneType
import numpy as np
from sklearn import neighbors

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#
class RobotSim:
    def __init__(self,stateModel:StateModel,trueState , observationModel:ObservationModel, transitionalModel: TransitionModel):
        self.__sm = stateModel
        self.trueState = trueState
        self.__om = observationModel
        self.__tm = transitionalModel

        self.gridRows, self.gridCols,self.Heading = self.__sm.get_grid_dimensions()         
    
    def move(self) -> int:

        """
        tempX,tempY,tempH = self.__sm.state_to_pose(self.trueState)

        headings = []
        if tempX > 0:
            headings.append(0)
        if tempX < self.gridRows-1:
            headings.append(2)
        if tempY > 0:
            headings.append(3)
        if tempY < self.gridCols-1:
            headings.append(1)

        if tempH in headings and rnd.random() < 0.3:
            headings.remove(tempH)
            tempH = rnd.choice(headings)
        else:
            tempH = rnd.choice(headings)
        
        if   tempH == 0: tempX -= 1
        elif tempH == 1: tempY += 1
        elif tempH == 2: tempX += 1
        elif tempH == 3: tempY -= 1

        self.trueState = self.__sm.pose_to_state(tempX,tempY,tempH)
        """
        current_state = self.trueState    # State
        
        next_state = -1 # Non-existant state to initialize with
        probs = []
        states = []
        for state in range(self.__tm.get_num_of_states()-1):
            probs.append(self.__tm.get_T_ij(current_state, state))        
            states.append(state)
            
        next_state = np.random.choice(states, 1, p=probs)[0]
        if next_state >= 0:
            current_state = next_state     # If still -1, then no good state found
        else:
            return None
        
        self.trueState = current_state
    
    def getState(self):
        return self.trueState
    
    def senseLoc(self):
        """
        prob = rnd.random()
        x,y = self.__sm.state_to_position(self.trueState)
        neigborsOne = self.getNeigbors(x,y,1)
        neighborsTwo = self.getNeigbors(x,y,2)
        #print(neigborsOne)
        #print(neighborsTwo)
        rnd.shuffle(neigborsOne)
        rnd.shuffle(neighborsTwo)
        if prob < 0.1:
            return x , y
        elif prob < 0.1 + 0.05*len(neigborsOne):
            return neigborsOne[0]
        elif prob < 0.1 + 0.05*len(neigborsOne) + 0.025*len(neighborsTwo):
            return neighborsTwo[0]
        else:
            return None
        """
        
        current_state = self.trueState     # State
        
        probs = []
        readings = []
        readings.append(None)
        probs.append(self.__om.get_o_reading_state(None, current_state))
        for reading in range(self.__om.get_nr_of_readings()-1):
            probs.append(self.__om.get_o_reading_state(reading, current_state))
            readings.append(reading)
        
        probs /= sum(probs)
        print("Readings: {}".format(readings))
        sensor = np.random.choice(readings, 1, p=probs)[0]
        
                  
        return sensor        
        


    def getNeigbors(self,x:int,y:int,neigborRank:int):
        if neigborRank == 2 :
            possible = \
                         [(x-2,y)   , (x+2,y)   , (x,y-2)    , 
                          (x,y+2)   , (x-2,y-2) , (x+2,y+2)  ,  
                          (x-2,y+2) , (x+2,y-2) , (x+2,y+1)  , 
                          (x+1,y+2) , (x-1, y-2), (x-2, y-1) , 
                          (x-1,y+2) , (x+1, y-2), (x-2, y+1 ), 
                          (x+2, y-1)                          
                         ]
        elif neigborRank == 1:
             possible = \
                            [(x-1,y)  , (x+1,y)  , (x,y-1)  ,
                             (x,y+1)  , (x-1,y-1), (x+1,y+1),
                             (x-1,y+1), (x+1,y-1) ]
        neigh = []
        for x_pos,y_pos in possible:
            if self.inBoard(x_pos,y_pos): neigh.append((x_pos,y_pos))

        """ Old method turned out not to work
        for nx in range(x - neigborRank, x + neigborRank + 1):
            for ny in range (y - neigborRank, y + neigborRank +1):
                if self.inBoard(nx, ny) and (nx != x and ny != y):
                    neigh.append((nx, ny))
        """
        return neigh
    
    def inBoard(self, x, y):
        return 0 <= x <= (self.gridCols -1)and 0 <= y <= (self.gridRows -1)

class HMMFilter:
    def __init__(self, stateModel :StateModel, transitionModel:TransitionModel, observationModel:ObservationModel):
        self.__stateModel = stateModel
        self.__observationModel = observationModel
        self.__transitionModel = transitionModel
        

    def update(self, senseReading, probabilities):

        diagonal = self.__observationModel.get_o_reading(senseReading)
        transpose = self.__transitionModel.get_T_transp()
        probabilities = diagonal @ transpose @ probabilities
        probabilities = 1/np.linalg.norm(probabilities) * probabilities
        
        return probabilities, np.argmax(probabilities)