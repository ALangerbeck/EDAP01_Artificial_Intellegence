
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
        # Changes direction randomly or if found a wall:
        tempX,tempY,tempH = self.__sm.state_to_pose(self.trueState)

        possibleMove = []
        if tempX > 0:
            possibleMove.append(0)
        if tempX < self.gridRows-1:
            possibleMove.append(2)
        if tempY > 0:
            possibleMove.append(3)
        if tempY < self.gridCols-1:
            possibleMove.append(1)

        if tempH in possibleMove and rnd.random() < 0.3:
            possibleMove.remove(tempH)
            tempH = rnd.choice(possibleMove)
        else:
            tempH = rnd.choice(possibleMove)
        
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
        closestNeigh = self.getClosestNeigh(x,y)
        secondNeigh = self.getSecondNeigh(x,y)
        rnd.shuffle(closestNeigh)
        rnd.shuffle(secondNeigh)
        if prob < 0.1:
            return x , y
        elif prob < 0.1 + 0.05*len(closestNeigh):
            return closestNeigh[0]
        elif prob < 0.1 + 0.05*len(closestNeigh) + 0.025*len(secondNeigh):
            return secondNeigh[0]
        else:
            return None
        
    def getClosestNeigh(self,x,y):
        neigh = []
        for nx in range(x - 1, x + 1):
            for ny in range (y - 1, y + 1):
                if self.inBoard(nx, ny) and (nx != x and ny != y):
                    neigh.append((nx, ny))
        return neigh

    def getSecondNeigh(self,x,y):
        neigh = []
        for nx in range(x - 2, x + 2):
            for ny in range (y - 2, y + 2):
                if self.inBoard(nx, ny) and (nx != x and ny != y):
                    neigh.append((nx, ny))
        return neigh
    
    def inBoard(self, x, y):
        return 0 <= x <= (self.gridCols -1)and 0 <= y <= (self.gridRows -1)
#
# Add your Filtering approach here (or within the Localiser, that is your choice!)
#
class HMMFilter:
    def __init__(self, stateModel :StateModel, transitionModel, observationModel):
        self.__stateModel = stateModel
        self.__observationModel = observationModel
        self.__transitionModel = transitionModel
        

    def update(self, reading, probabilities):
        if reading :
            senseReading = self.__stateModel.position_to_reading(reading[0], reading[1])
        else:
            senseReading = None

        probabilities = np.matmul(np.matmul(self.__observationModel.get_o_reading(senseReading), self.__transitionModel.get_T_transp()), probabilities)       
        probabilities = (1.0 / sum(probabilities) ) * probabilities
        # Updates the values of the f vector based on the forward algorithm:
        # Om should be diag matrix, and check if reading = none
        TempProbabilities = {}
        for i,j in enumerate(probabilities):
            cords = self.__stateModel.state_to_position(i)
            TempProbabilities[cords] = TempProbabilities.get(cords,0) + j
        
        return probabilities, max(TempProbabilities,key=TempProbabilities.get)

    def filtering(self, sense, probs):
        #Get sensed position as reading
        senseReading = self.__stateModel.position_to_reading(sense[0], sense[1]) if sense else None

        T_trans = self.__transitionModel.get_T_transp()
        O = self.__observationModel.get_o_reading(senseReading)
        probs = np.matmul(np.matmul(O, T_trans), probs)       
        
        probs = (1.0 / sum(probs) ) * probs         #Normalize
        
        #print most likely states for debugging
        # topStates = (-probs).argsort()[:5]
        # for s in topStates:
        #     print("Pose:", self.sm.state_to_pose(s), "Likelihood:", res[s])

        #estimate = self.sm.state_to_position(np.argmax(res))

        #sum probabilitites of states corresponding to the same position 
        estimate = self.getEstimate(probs)
        return probs, estimate
    
        
        
        
