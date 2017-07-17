import h5py
import pickle
import numpy as np
import sys
from keras.models import load_model
import simulator
import os
import project
import importlib

model=load_model(os.path.join(project.modeldir,"model_1.h5"))
print(model.summary())
ninputs=len(model.input_shape)

sim=simulator.Simulator()
config=sim.connect({"trackname":project.trackname,'controller':importlib.util.find_spec("EmbeddedController").origin})


while True:
    state=sim.get_state()
    p=model.predict(project.State2X(state)[:ninputs])
    steering=p[0][0,0]
    throttle=p[1][0,0]
    print("steering {:5.3f} throttle {:5.3f} speed={:5.4f}".format(steering,throttle,state["speed"]))
    throttle=0.8 # disable throttle control
    sim.send_cmd({"steering":steering,'throttle':throttle})