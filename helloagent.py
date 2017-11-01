import random, itertools
from hfo import *
import csv
import numpy as np
import theano
import theano.tensor as T
import cPickle
import lasagne
import os
import sys



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

def saveGoldAcions(state,action,filename):

	with open("".join(('aug_states', filename,".csv")),'a') as f_handle:
	   np.savetxt(f_handle,state, newline=",")
	   f_handle.write("\n")

	with open("".join(('aug_actions', filename,".csv")),'a') as f_handle:
	   np.savetxt(f_handle,action, newline=",")
	   f_handle.write("\n")

def main():
  
  pklfile=  sys.argv[1]
  print pklfile
  augment_Flag = (sys.argv[2]=='true')
  lstm_Flag = (sys.argv[3]=='true')
  datawritename = sys.argv[4]

  #print pklfile,augment_Flag,lstm_Flag,datawritename
  #os.remove('foo.csv')
  fv=featureVector()
  #fv.printfv()
  #for attr in fv.__dict__.iteritems():
  #      print attr
  if pklfile == "0":
	pklfile = "E_agentno shuffle.pkl"
  else:
	#pklfile = "".join(('E_agent',pklfile,'.pkl'))
	pklfile = "".join(('E_agentAug',pklfile,'.pkl'))
  handle = open(pklfile, 'r')
  network = cPickle.load(handle)
  #print type(network)
  #print np.shape(network[0])
  states = np.zeros(1)

  #input_var = T.fvector('inputs')
  prediction = lasagne.layers.get_output(network)
  inputlayer = lasagne.layers.get_all_layers(network)[0]
  predict_fn = theano.function([inputlayer.input_var], prediction)

  #print "##############"

  # Create the HFO Environment
  hfo = HFOEnvironment() 
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(LOW_LEVEL_FEATURE_SET,
                      '../bin/teams/base/config/formations-dt', 6000,
                      'localhost', 'base_left', False)
  for episode in itertools.count():
    status = IN_GAME
    states = np.zeros(1)
    while status == IN_GAME:
      # Get the vector of state features for the current state
      state = hfo.getState()

      fv.fill(state)
      state=state.reshape(-1,66)
      #state=state.reshape(-1,1,66)
      if states.size>10000000000000000:
	#print "WRONG"
      	states = np.concatenate((states,state),axis=0)
        if np.shape(states)[0]>50:
	   states=np.delete(states, 0, 0)
      else:
	states= state

      if lstm_Flag:
	seq = states.reshape(1,-1,66)
      else:
	seq = states.reshape(-1,66)
      #print np.shape(seq)
      rng = np.random.RandomState()

      #########################################
      #JUST SAVING DATA ########################
      if augment_Flag:
	#print "SAVING"
	actionVector = np.zeros(6)
	if fv.kickable == 1 :
		if fv.goal_center_prox > 0.5:
		  angle = np.arccos(fv.goal_post_top_cos)*np.sign(fv.goal_post_top_sin)
		  actionVector[4]=1
		  actionVector[5]=np.degrees(angle)/180
		else:
		  angle = np.arccos(fv.goal_center_cos)*np.sign(fv.goal_center_sin)
		  actionVector[4]=0.2
		  actionVector[5]=np.degrees(angle)/180
	else:
	  angle = np.arccos(fv.ball_angle_cos)*np.sign(fv.ball_angle_sin)
	  actionVector[0]=1
	  actionVector[1]=np.degrees(angle)/180

	saveGoldAcions(state[0],actionVector,datawritename)

      ##########################################


      if augment_Flag and rng.rand()<0.5:# if HANDCRAFTED
##############################################
	#print "HANDCRAFTED"
	if fv.kickable == 1 :
	  if fv.goal_center_prox > 0.5:
	  	angle = np.arccos(fv.goal_post_top_cos)*np.sign(fv.goal_post_top_sin)
		hfo.act(KICK, 100, np.degrees(angle))
	  else:
	  	angle = np.arccos(fv.goal_center_cos)*np.sign(fv.goal_center_sin)
		hfo.act(KICK, 20, np.degrees(angle))
	else:
	  	angle = np.arccos(fv.ball_angle_cos)*np.sign(fv.ball_angle_sin)
		hfo.act(DASH, 100, np.degrees(angle))
################################################
      else:# Prediction
	prediction = predict_fn(seq)
	#prsingle = predict_fn(state.reshape(-1,1,66))
        #print prediction
	prediction = prediction[-1]
        #print prsingle
	#print "#############################################"
	maxindex=prediction.argmax()

	if maxindex <2:
		hfo.act(DASH, prediction[0]*100, prediction[1]*180)
		#print "DASH ", prediction[0]*100, ",", prediction[1]*180
	elif maxindex == 2:
		hfo.act(TURN,prediction[2]*180)
		#print "TURN ", prediction[2]*180
	elif maxindex == 3:
		hfo.act(TACKLE,prediction[3]*180)
		#print "TACKLE ", prediction[3]*180
	else:
		hfo.act(KICK, prediction[4]*100, prediction[5]*180)
		#print "KICK ", prediction[4]*100, ",", prediction[5]*180
      #print type(state)
      
      #writecsv(state)
      #print state
      #fv.printfv()
      # Perform the action
      #hfo.act(TURN,5)
      #hfo.act(DASH, 1000, 0)
      # Advance the environment and get the game status
      status = hfo.step()

    if status == SERVER_DOWN:
      hfo.act(QUIT)
      break

if __name__ == '__main__':
  main()
