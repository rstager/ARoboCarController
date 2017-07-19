import h5py
import pickle
import numpy as np
import sys
import random
import arobocar
import os
import project
import importlib
import gym


# This controller just follows the PID recommendations most of the time but deviates to capture off-policy state
# this controller also records state

# runtime configuration parameters
# steering_noise - amount of noise to add to steering and throttle
# noise_probabilit - how often to deviate - set to zero to drive correctly
# deviation_duration - duration of deviation
def run(project,env,maxidx, filename="robocar.hdf5",
        steering_noise=.15,
        throttle_noise=.15,
        noise_probability=0.01,
        deviation_duration=40 ):

    fullpathname=os.path.join(project.datadir,filename) # or None

    #now open the h5 file
    output = h5py.File(fullpathname, 'w')
    recorder=project.recorder(output,project.config,maxidx)
    actions = output.create_dataset('actions', (maxidx, 2))
    actions.attrs['cols'] = "steering,throttle"

    steering=0
    throttle=0
    deviating_cnt=0

    #get started
    observation=env.reset()


    for h5idx in range(0,maxidx):
        # get images and state from simulatorq
        observation,reward,done,info = env.step([steering, throttle])

        if(done):
            print("Done")
            recorder.record(observation, reward, done, info)
            env.reset()
            continue

        #use the PID values by default
        steering=info["PIDsteering"]
        throttle=info["PIDthrottle"]
        offset=info["pathoffset"] #distance from center of road
        speed=observation[1][0]
        idx=recorder.record(observation, reward, done, info)
        actions[idx]=np.array([steering, throttle])

        if deviating_cnt > 0 and ( abs(offset) > 75 or speed>1600) :  # stop deviating if we ran off the road or go too fast
            deviating_cnt = 0
            print("Abort deviation")

        if deviating_cnt>0: # while deviation
            if deviation_type == 0:
                steering = deviation_angle
            else:
                throttle = deviation_throttle
            deviating_cnt -= 1
            if (deviating_cnt == 0):
                print("End deviation")

        #decide when to start another deviation
        if deviating_cnt == 0 and random.random() < noise_probability:
            deviating_cnt = deviation_duration
            if(random.random()>0.5):
                deviation_angle = steering + random.random() * steering_noise - (steering_noise / 2)
                deviation_type=0
                print("** Begin Steering deviation {}".format(deviation_angle))
            else:
                deviation_throttle = throttle + random.random() * throttle_noise - (throttle_noise / 2)
                deviation_type=1
                print("** Begin Throttle deviation {}".format(deviation_throttle))

    output.close()

if __name__ == "__main__":
    # connect to environment
    env = gym.make('ARoboCar-v0')
    run( project, env, 10000)