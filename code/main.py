import os
import time
from subprocess import call

import numpy as np
from tqdm import tqdm

from env.backgammon import BackgammonEnv
from network import Network
from agent import TDAgent

from WallTime import WallTimeWatcher

###############################################################################################################################################################
# TRAIN THE TDAGENT WITH THE FOLLOWING VALUES
# 0 , int(5e4) 
# 10 , int(5e4)
# 20 , int(1e5)
# 40 . int(4e5)

hidden_units, max_episodes = 0, int(5e4)

###############################################################################################################################################################
# AGENT STUFF  

# initiate the environment
env = BackgammonEnv()

# build the model
network = Network(hidden_units)
network.build([1,198])

# inititate the agent
alpha = 0.1
_lambda = 0.7
gamma = 1
agent = TDAgent(env, network, alpha, _lambda, gamma)

###############################################################################################################################################################
# STUFF THAT NEEDS DOING IF THE JOB HAS BEEN RESUBMITTED

# get number of previously completed episodes
if os.path.exists('completed_episodes.npy'): # make sure to delete this file between agents!!!!!!!
    _file = np.load('completed_episodes.npy')
    completed_episodes = _file[0]
else:
    completed_episodes = 0

episodes2go = max_episodes-completed_episodes

# if the wall time has previously run out, load the most recent checkpoint
if completed_episodes > 0:
    filepath = 'TDAgent'+str(hidden_units)+'episode'+str(completed_episodes)+'of'+str(max_episodes)+'.hdf5'
    print('filepath: '+filepath)
    network.load_weights(filepath)

###############################################################################################################################################################
# WALL TIME STUFF
# WE GET THE TIME LAST SO THAT OUR START TIME IS ACCURATE

# get the wall time
walltimeString = str(os.environ['WALL'])

# method for translating wall time into seconds
def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# how many seconds of wall time do we have?
walltime = get_sec(walltimeString)
# what's the current time?
start_time = time.time()

# create watcher object
watch = WallTimeWatcher(start_time,walltime)

###############################################################################################################################################################
# LEARNING AND RESUBMISSION STUFF

for i in tqdm (range(episodes2go), desc = "Learning..."):
    watch.on_episode_begin()
    agent.learn()
    completed_episodes += 1
    # check if our wall time might run out next episode
    reachedwalltime = watch.on_episode_end() 
    if reachedwalltime:
        # create a checkpoint
        network.save_weights('TDAgent'+str(hidden_units)+'episode'+str(completed_episodes)+'of'+str(max_episodes)+'.hdf5')
        # export number of completed episodes to a file
        n = [completed_episodes]
        np.save('completed_episodes.npy',n)
        # resubmit the job
        call('qsub job.sge', shell=True)
        break

