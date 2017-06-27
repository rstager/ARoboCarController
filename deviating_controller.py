import h5py
import pickle
import numpy as np
import sys
import random
import simulator
import project
import os

# This controller just follows the PID recommendations most of the time but deviates to capture off-policy state
# this controller also records state

#runtime configuration parameters
filename=os.path.join(project.datadir,"robocar.hdf5") # or None
steering_noise=.15      #amount of noise to add to steering
noise_probability=0.01  #how often to deviate - set to zero to drive correctly
deviation_duration=40   # duration of deviation

sim=simulator.Simulator()
config=sim.connect()
print (config)
height=config["cameraheight"]
width=config["camerawidth"]

#now open the h5 file
maxidx=32
output = h5py.File(filename, 'w')
images = output.create_dataset('frontcamera', (maxidx, height, width, 3), 'i1',
                                         maxshape=(None, height, width, 3))
images.attrs['description'] = "simple test"


controls = output.create_dataset('steering.throttle', (maxidx, 2), maxshape=(None, 2))

#parameters for deviating
deviating_cnt=0
h5idx=0
while True:
  try:
    # get images and state from simulator
    # record images and steering,throttle
    state=sim.get_state()
    controls[h5idx] = [state["PIDsteering"], state["PIDthrottle"]]
    images[h5idx] = state["frontcamera"]

    h5idx += 1
    if(h5idx>=maxidx):
        maxidx += 32
        images.resize((maxidx, height, width, 3))
        controls.resize((maxidx, 2))
        output.flush()
        print("Flushing h5")

    #print("pathdistance {:7f} offset {:5f} PID {:7f}  {:5.3f} dt={:5.4f}".format(state["pathdistance"], state["offset"], state["PIDthrottle"], state["PIDsteering"],state["delta_time"]))
    #use the PID values by default
    steering=state["PIDsteering"]
    throttle=state["PIDthrottle"]
    offset=state["pathoffset"] #distance from center of road

    if deviating_cnt > 0 and abs(offset) > 75:  # stop deviating if we ran off the road
        deviating_cnt = 0
        print("Abort deviation")

    if deviating_cnt>0: # while deviation
        steering = deviation_angle
        deviating_cnt -= 1
        if (deviating_cnt == 0):
            print("End deviation")

    #decide when to start another deviation
    if deviating_cnt == 0 and random.random() < noise_probability:
        deviating_cnt = deviation_duration
        deviation_angle = steering + random.random() * steering_noise - (steering_noise / 2)
        print("** Begin Steering deviation {}".format(deviation_angle))

    sim.send_cmd({"steering":steering,'throttle':throttle})

  except TypeError:
    pass
