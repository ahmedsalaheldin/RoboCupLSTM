import random, itertools
from hfo import *
import csv
import numpy as np
import theano
import theano.tensor as T
import cPickle
import lasagne
import os
import random




class featureVector:
    def __init__(self):
        self.self_pos_valid = 0
        self.self_vel_valid  = 0
	self.self_vel_ang_sin = 0
	self.self_vel_ang_cos = 0
	self.self_vel_mag = 0
	self.self_ang_sin = 0
	self.self_ang_cos = 0
	self.stamina = 0
	self.frozen = 0
	self.colliding_with_ball = 0
	self.colliding_with_player = 0
	self.colliding_with_post = 0
	self.kickable = 0
	self.goal_center_sin = 0
	self.goal_center_cos = 0
	self.goal_center_prox = 0
	self.goal_post_top_sin = 0
	self.goal_post_top_cos = 0
	self.goal_post_top_prox = 0
	self.goal_post_bot_sin = 0
	self.goal_post_bot_cos = 0
	self.goal_post_bot_prox = 0
	self.pen_box_cen_sin = 0
	self.pen_box_cen_cos = 0
	self.pen_box_cen_prox = 0
 	self.pen_box_top_sin = 0
	self.pen_box_top_cos = 0
	self.pen_box_top_prox = 0
 	self.pen_box_bot_sin = 0
	self.pen_box_bot_cos = 0
	self.pen_box_bot_prox = 0
	self.center_field_sin = 0
	self.center_field_cos = 0
	self.center_field_prox = 0
	self.corner_top_left_sin = 0
	self.corner_top_left_cos = 0
	self.corner_top_left_prox = 0
	self.corner_top_right_sin = 0
	self.corner_top_right_cos = 0
	self.corner_top_right_prox = 0
	self.corner_bot_right_sin = 0
	self.corner_bot_right_cos = 0
	self.corner_bot_right_prox = 0
	self.corner_bot_left_sin = 0
	self.corner_bot_left_cos = 0
	self.corner_bot_left_prox = 0
	self.oob_left_dist = 0
	self.oob_right_dist = 0
	self.oob_top_dist = 0
	self.oob_bot_dist = 0
	self.ball_pos_valid = 0
	self.ball_angle_sin = 0
	self.ball_angle_cos = 0
	self.ball_dist = 0
	self.ball_vel_valid = 0
	self.ball_vel_mag = 0
	self.ball_vel_ang_sin = 0
	self.ball_vel_ang_cos = 0

    def fill(self,status):
	st=iter(status)
	self.self_pos_valid = next(st)             #0
	self.self_vel_valid  = next(st)		   #1
  	self.self_vel_ang_sin = next(st)	   #2
	self.self_vel_ang_cos = next(st)	   #3
	self.self_vel_mag = next(st)		   #4
	self.self_ang_sin = next(st)		   #5
	self.self_ang_cos = next(st)  	 	   #6
	self.stamina = next(st)	                   #7
	self.frozen = next(st) 			   #8
	self.colliding_with_ball = next(st) 	   #9
	self.colliding_with_player = next(st)      #10
	self.colliding_with_post = next(st) 	   #11
	self.kickable = next(st)		   #12
	self.goal_center_sin = next(st)		   #13
	self.goal_center_cos = next(st)		   #14
	self.goal_center_prox = next(st)	   #15
	self.goal_post_top_sin = next(st)	   #16
	self.goal_post_top_cos = next(st)	   #17
	self.goal_post_top_prox = next(st)	   #18
 	self.goal_post_bot_sin = next(st)	   #19
	self.goal_post_bot_cos = next(st)	   #20
	self.goal_post_bot_prox = next(st)	   #21
 	self.pen_box_cen_sin = next(st) 	   #22
	self.pen_box_cen_cos = next(st) 	   #23
	self.pen_box_cen_prox = next(st)	   #24
 	self.pen_box_top_sin = next(st)		   #25
	self.pen_box_top_cos = next(st)	  	   #26
	self.pen_box_top_prox = next(st)	   #27
 	self.pen_box_bot_sin = next(st)		   #28
	self.pen_box_bot_cos = next(st)		   #29
	self.pen_box_bot_prox = next(st)	   #30
	self.center_field_sin = next(st)	   #31
	self.center_field_cos = next(st)	   #32
	self.center_field_prox = next(st)	   #33
	self.corner_top_left_sin = next(st)	   #34
	self.corner_top_left_cos = next(st)	   #35
	self.corner_top_left_prox = next(st)	   #36
	self.corner_top_right_sin = next(st)	   #37
	self.corner_top_right_cos = next(st)	   #38
	self.corner_top_right_prox = next(st)	   #39
	self.corner_bot_right_sin = next(st)       #40
	self.corner_bot_right_cos = next(st)       #41
	self.corner_bot_right_prox = next(st)	   #42
	self.corner_bot_left_sin = next(st)	   #43
	self.corner_bot_left_cos = next(st)        #44
	self.corner_bot_left_prox = next(st)       #45
	self.oob_left_dist = next(st)		   #46
	self.oob_right_dist = next(st)		   #47
	self.oob_top_dist = next(st)		   #48
	self.oob_bot_dist = next(st)		   #49
	self.ball_pos_valid = next(st)		   #50
	self.ball_angle_sin = next(st)	   	   #51
	self.ball_angle_cos = next(st)		   #52
	self.ball_dist = next(st)		   #53
	self.ball_vel_valid = next(st)		   #54
	self.ball_vel_mag = next(st)		   #55
	self.ball_vel_ang_sin = next(st)	   #56
	self.ball_vel_ang_cos = next(st)	   #57
	
    def printfv(self):	
	for attr , val in self.__dict__.iteritems():
            print attr, val


def writecsv(state):

	with open("foo.csv",'a') as f_handle:
	   np.savetxt(f_handle,state, newline=",")
	   f_handle.write("\n")
	#np.savetxt("foo.csv", state, delimiter=",")
	#with open('features.csv', 'wb') as csvfile:
	#    spamwriter = csv.writer(csvfile, delimiter=' ',
	#	                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
	#    spamwriter.writerow(state)

def saveGoldAcions(state,action):
	with open("E_LSTMstates.csv",'a') as f_handle:
	   np.savetxt(f_handle,state, newline=",")
	   f_handle.write("\n")

	with open("E_LSTMactions.csv",'a') as f_handle:
	   np.savetxt(f_handle,action, newline=",")
	   f_handle.write("\n")

def move(targetdist, targetang,hfo):
  distance = (1-targetdist)*100
  hfo.act(DASH, 100, np.degrees(targetang))
  
def kickToGoal(targetdist, targetang,hfo):
  hfo.act(KICK, targetdist, np.degrees(targetang))
  #print targetang


  

def main():


  fv=featureVector()

  # Create the HFO Environment
  hfo = HFOEnvironment() 
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                      '../bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)
  count =0
  for episode in itertools.count():
    status = IN_GAME   
    count =0
    while status == IN_GAME:
      # Get the vector of state features for the current state
      state = hfo.getState()
      fv.fill(state)

      actionVector = np.zeros(6)
      if fv.kickable == 1 :
	 if fv.goal_center_prox > 0.5:
	  	angle = np.arccos(fv.goal_post_top_cos)*np.sign(fv.goal_post_top_sin)
		hfo.act(KICK, 100, np.degrees(angle))
	        actionVector[4]=1
	        actionVector[5]=np.degrees(angle)/180
	 else:
	  	angle = np.arccos(fv.goal_center_cos)*np.sign(fv.goal_center_sin)
		hfo.act(KICK, 20, np.degrees(angle))
	        actionVector[4]=0.2
	        actionVector[5]=np.degrees(angle)/180
      else:
	 angle = np.arccos(fv.ball_angle_cos)*np.sign(fv.ball_angle_sin)
	 hfo.act(DASH, 100, np.degrees(angle))
	 actionVector[0]=1
	 actionVector[1]=np.degrees(angle)/180

      status = hfo.step()
      count = count+1
      saveGoldAcions(state,actionVector)
      if status != IN_GAME:
	print "count = ",count
	for i in range(150-count):
		saveGoldAcions(np.zeros(66),np.zeros(6))
      #print status

    if status == SERVER_DOWN:
      hfo.act(QUIT)
      break

if __name__ == '__main__':
  main()
