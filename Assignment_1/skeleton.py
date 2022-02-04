from curses import window
from email import message
from os import lseek
from pickle import FALSE, TRUE
from sqlite3 import Row
from tkinter.tix import Tree
from unittest import case
import gym
import random
import requests
import numpy as np
import argparse
import sys
import copy
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")
SEARCH_TREE_MAX_DEPTH = 5
DEBUG = True

COLUMN_COUNT = 7
ROW_COUNT = 6

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["ELITE"] # TODO: fill this list with your stil-id's

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   print(list(avmoves))
   
   action = random.choice(list(avmoves))
   
   #action = int(input ("Enter a number: "))
   
   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

def student_move(env:ConnectFourEnv):
   """ 
   TODO: Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   """
   move,value = minmax(env,None,None,0,-np.inf,np.inf,True)   
   
   if DEBUG:
         print("Value of chosen move") 
         print(value)
   return move
   
   #return random.choice(list(env.available_moves()))




def EvaluateBoard(state:np.ndarray,max_player:bool):
   
  
   if max_player:
      badPiece = -1
   else:
      badPiece = 1

   score = 3*7 + 4*6 + 12 +12
   badscore = 3*7 + 4*6 + 12 +12

   #row socring
   for i in range(ROW_COUNT):
      for j in range(COLUMN_COUNT - 3):
            window = state[i][j:j + 4]
            if np.count_nonzero(window == badPiece) > 0:
               score -= 1
            if np.count_nonzero(window == -badPiece) > 0:
               badscore -= 1
   
   # Test columns on transpose array
   reversed_board = [list(i) for i in zip(*state)]
   for i in range(COLUMN_COUNT):
      for j in range(ROW_COUNT - 3):
            window = reversed_board[i][j:j + 4]
            if np.count_nonzero(window == badPiece) > 0:
               score -= 1
            if np.count_nonzero(window == -badPiece) > 0:
               badscore -= 1
   
    # Test diagonal
   for i in range(ROW_COUNT - 3):
      for j in range(COLUMN_COUNT - 3):
            badPieceCount = 0
            pieceCount = 0
            for k in range(4):
               badPieceCount += np.count_nonzero(state[i + k][j + k] == badPiece)
               if badPieceCount > 0:
                  score -= 1
               pieceCount += np.count_nonzero(state[i + k][j + k] == -badPiece)
               if pieceCount > 0:
                  badscore -= 1

   reversed_board = np.fliplr(state)
   # Test reverse diagonal
   for i in range(ROW_COUNT - 3):
      for j in range(COLUMN_COUNT - 3):
            badPieceCount = 0
            pieceCount = 0
            for k in range(4):
               badPieceCount += np.count_nonzero(reversed_board[i + k][j + k] == badPiece)
               if badPieceCount > 0:
                  score -= 1
               pieceCount += np.count_nonzero(reversed_board[i + k][j + k] == -badPiece)
               if pieceCount > 0:
                  badscore -= 1
   
   return score - badscore
   """
   
   eval_weights = [[3, 4, 5, 7, 5, 4, 3],
                    [4, 6, 8, 10, 8, 6, 4],
                    [5, 8, 11, 13, 11, 8, 5],
                    [5, 8, 11, 13, 11, 8, 5],
                    [4, 6, 8, 10, 8, 6, 4],
                    [3, 4, 5, 7, 5, 4, 3]]

   utility = 0
   for i in range(len(eval_weights)):
      for j in range(len(eval_weights[i])):
         if state[i][j] == 1:
                utility += eval_weights[i][j]
         elif state[i][j] == -1:
                utility -= eval_weights[i][j]

   return utility
   """

def minmax(env:ConnectFourEnv,reward:int,state:np.ndarray, depth,alpha,beta, max_player):
   alphaLocal = alpha
   betaLocal = beta
   
   #print("reward:")
   #print(reward)

   if (reward == 1) and max_player: 
      #if DEBUG: print("i see a winning move")
      #return 110* (SEARCH_TREE_MAX_DEPTH - depth)
      return None,10000000000
   elif (reward == 1) and (not max_player):
      #if DEBUG: print("i see a losing move")
      #return  -100 * (SEARCH_TREE_MAX_DEPTH - depth)
      return None,-np.inf
      
   elif (reward == 0.5):
      return None,-10000000000
   elif depth == SEARCH_TREE_MAX_DEPTH:
         return None,EvaluateBoard(state,max_player)
      
   
   avmoves = list(env.available_moves())

   if max_player:
      value = -np.inf
      move = random.choice(avmoves)
      for x in avmoves:
         next_env = copy.deepcopy(env)
         state, reward, done, _ = next_env.step(x)
         temp_value = minmax(next_env,reward,state,depth + 1,alphaLocal,betaLocal, False)[1]
         if(temp_value > value):
            value = temp_value
            move = x
         alphaLocal = max(alphaLocal,value)
         if alphaLocal >= betaLocal:
            break #beta cutoff
      return move,value   
   else: #min_player
      value = np.inf
      move = random.choice(avmoves)
      for x in avmoves:
         next_env = copy.deepcopy(env)
         state, reward, done, _ = next_env.step(x)
         temp_value = minmax(next_env,reward,state,depth + 1,alphaLocal,betaLocal, True)[1]
         if(temp_value < value):
            value = temp_value
            move = x
         betaLocal = min(betaLocal,value)
         if betaLocal <= alphaLocal:
               break #alpha cutoff
      return move,value
      
def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      env.reset(board=None)
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss
      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      print(botmove)
      state = np.array(res.json()['state'])
      
      if botmove != -1:
         env.change_player()
         env.step(botmove)
         env.change_player()

   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(env) # TODO: change input here

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         
         res = call_server(stmove)
         print(res.json()['msg'])

         env.step(stmove)
         env.change_player()

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])


         print("server move:")
         print(botmove)
         print("bot move: ")
         print (stmove)

         if botmove != -1:
            board,_,_,_ =  env.step(botmove)
            env.change_player()

      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tied to make an illegal move! Games ends.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   parser.add_argument("-t", "--ten", help = "Plays online games until total reward is at least 10", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.ten:
      stats = check_stats()
      while(stats['streak'] < 20):
         play_game(vs_server = True)
         stats = check_stats()
         print("Streak = ")
         print(stats['streak'])
      print("total streak should now be at least 20")

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
