
from os import remove
import random as rnd
from types import NoneType
import numpy as np

from models import TransitionModel,ObservationModel,StateModel

#
# Add your Robot Simulator here
#
class RobotSim:
    def __init__(self,stateModel:StateModel,trueState):
        self.__sm = stateModel
        self.trueState = trueState

        self.gridRows, self.gridCols,self.Heading = self.__sm.get_grid_dimensions()         
    
    def move(self) -> int:
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
    
    def getState(self):
        return self.trueState
    
    def senseLoc(self):
        prob = rnd.random()
        x,y = self.__sm.state_to_position(self.trueState)
        neigborsOne = self.getNeigbors(x,y,1)
        neighborsTwo = self.getNeigbors(x,y,2)
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
        
    def getNeigbors(self,x:int,y:int,neigborRank:int):
        neigh = []
        for nx in range(x - neigborRank, x + neigborRank + 1):
            for ny in range (y - neigborRank, y + neigborRank +1):
                if self.inBoard(nx, ny) and (nx != x and ny != y):
                    neigh.append((nx, ny))
        return neigh
    
    def inBoard(self, x, y):
        return 0 <= x <= (self.gridCols -1)and 0 <= y <= (self.gridRows -1)

class HMMFilter:
    def __init__(self, stateModel :StateModel, transitionModel:TransitionModel, observationModel:ObservationModel):
        self.__stateModel = stateModel
        self.__observationModel = observationModel
        self.__transitionModel = transitionModel
        

    def update(self, reading, probabilities):
        if reading :
            senseReading = self.__stateModel.position_to_reading(reading[0], reading[1])
        else:
            senseReading = None

        diagonal = self.__observationModel.get_o_reading(senseReading)
        transpose = self.__transitionModel.get_T_transp()
        probabilities = ((diagonal @ transpose) @ probabilities)/ np.linalg.norm((diagonal @ transpose) @ probabilities)
        
        return probabilities, np.argmax(probabilities)
        
