import sys
import os
import numpy as np
import csv

def setglobal():
	global prevframeF
	prevframeF=-1
	global prevframeS
	prevframeS=-1
	global prevframeA
	prevframeA=-1
	global featurelist
	featurelist = []
def writestatecsv(state):

	with open("states_big2.csv",'a') as f_handle:
	   np.savetxt(f_handle,state, newline=",")
	   f_handle.write("\n")
	#np.savetxt("foo.csv", state, delimiter=",")
	#with open('features.csv', 'wb') as csvfile:
	#    spamwriter = csv.writer(csvfile, delimiter=' ',
	#	                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
	#    spamwriter.writerow(state)
def writeactioncsv(action):

	with open("actions_big2.csv",'a') as f_handle:
	   np.savetxt(f_handle,action, newline=",")
	   f_handle.write("\n")
	#np.savetxt("foo.csv", state, delimiter=",")
	#with open('features.csv', 'wb') as csvfile:
	#    spamwriter = csv.writer(csvfile, delimiter=' ',
	#	                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
	#    spamwriter.writerow(state)

def parseFeatuers(linelist):
	#featureVector = np.ones(66) 
	featureVector = np.asarray(linelist[4:],dtype=np.float32)	
	#for target, source in zip(featureVector,linelist):
	#	target=source
	writestatecsv(featureVector)
		

def parseGameStatus(linelist):
	gameStatus= linelist[4]
	#IN_GAME, GOAL, CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME, SERVER_DOWN

def parsActions(linelist):
	actionVector = np.zeros(6)
	# remove closed bracket
	token = linelist[4].split(')')
	#get action type
	action = token[0].split('(')
	with open("actionindex.csv",'a') as csvfile:
		spamwriter = csv.writer(csvfile,delimiter=",")
		spamwriter.writerow([int(linelist[0])])
	if action[0] == 'Dash':
	     actionVector[0], actionVector[1] = action[1].split(',')
	     actionVector[0]=actionVector[0]/100
	     actionVector[1]=actionVector[1]/180
	elif action[0] == 'Turn':
	     actionVector[2] = action[1]
	     actionVector[2] = actionVector[2]/180
	elif action[0] == 'Tackle':
	     actionVector[3] = action[1].split(',')[0]
	     actionVector[3] = actionVector[3]/180
	elif action[0] == 'Kick':
	     actionVector[4], actionVector[5] = action[1].split(',')	
	     actionVector[4]=actionVector[4]/100
	     actionVector[5]=actionVector[5]/180
	else:
	     print action[0]
	     raise ValueError('Unrecognized action.')
	writeactioncsv(actionVector)

def categorizeLine(linelist):
#decide if line is status or state or action
 global prevframeF
 global prevframeS
 global prevframeA
 global featurelist

 if linelist[0] == '0':
	return

 if linelist[3] == 'StateFeatures':
	frameF = linelist[0]
	if frameF != prevframeF:
		#print frameF
		prevframeF = frameF
		featurelist=linelist
		#parseFeatuers(linelist)

 elif linelist[3] == 'GameStatus':
	frameS = linelist[0]
	if frameS != prevframeS:
		prevframeS = frameS
		parseGameStatus(linelist)
 else:
	frameA = linelist[0]
	if frameA != prevframeA:
		prevframeA = frameA
		if prevframeA==prevframeF:
			parsActions(linelist)
			parseFeatuers(featurelist)

#################################################
filename= str(sys.argv[1])
logdirectory= ""
logfile = open(logdirectory+filename, 'r')
setglobal()

count=1
for line in logfile:
	categorizeLine(line.split())
	count+=1

print count
