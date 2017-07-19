import h5py
import pickle
import numpy as np
import sys
import arobocar
import project
import gym


# This controller just follows the PID recommendations
env = gym.make('ARoboCar-v0')
observation=env.reset()
info={"PIDsteering":0,"PIDthrottle":0}


while True:

    #print(" throttle {:1.3f} speed {:1.3f} dt={:5.4f}".format( state["throttle"], state["speed"],state["delta_time"]))

    #print("pathdistance {:7.0f} offset {:5.0f} throttle {:1.3f} speed {:1.3f} angle {:1.3f} dt={:5.4f}".format(state["pathdistance"], state["pathoffset"], state["PIDthrottle"], state["PIDspeed"],state["PIDsteering"],state["delta_time"]))
    observation,reward,done,info =env.step([info['PIDsteering'], info['PIDthrottle']])
    if(done):
        print("Done")
        env.reset()
    else:
        print("observation={} reward={} done={} info={}".format(observation[1:], reward, done, info))