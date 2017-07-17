import h5py
import pickle
import numpy as np
import sys
import random
import simulator
import os
import project
import importlib


# This controller just follows the PID recommendations most of the time but deviates to capture off-policy state
# this controller also records state

#runtime configuration parameters
filename=os.path.join(project.datadir,"robocar.hdf5") # or None
steering_noise=.15      #amount of noise to add to steering
throttle_noise=.15
noise_probability=0.01  #how often to deviate - set to zero to drive correctly
deviation_duration=40   # duration of deviation

sim=simulator.Simulator()
config=sim.connect(project.connection_properties)
height=config["cameraheight"]
width=config["camerawidth"]

#now open the h5 file
maxidx=100000
output = h5py.File(filename, 'w')
datasets=project.createDatasets(config,output,maxidx)
controls = output.create_dataset('controls', (maxidx, 2))

#parameters for deviating
deviating_cnt=0

for h5idx in range(0,maxidx):
    # get images and state from simulatorq
    # record images and steering,throttle
    state=sim.get_state()
    controls[h5idx] = [state["PIDsteering"], state["PIDthrottle"]]
    for ds,x in zip(datasets,project.State2X(state)):  ds[h5idx]=x
    h5idx += 1
    output.flush()

    #print("pathdistance {:7f} offset {:5f} PID {:7f}  {:5.3f} dt={:5.4f}".format(state["pathdistance"], state["offset"], state["PIDthrottle"], state["PIDsteering"],state["delta_time"]))
    #use the PID values by default
    steering=state["PIDsteering"]
    throttle=state["PIDthrottle"]
    offset=state["pathoffset"] #distance from center of road
    speed=state['speed']

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


    sim.send_cmd({"steering":steering,'throttle':throttle})
