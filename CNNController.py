import h5py
import pickle
import numpy as np
import sys
from keras.models import load_model
import simulator
import os
import project

model=load_model(os.path.join(project.modeldir,"model_1.h5"))
print(model.summary())

sim=simulator.Simulator()
config=sim.connect({"trackname":project.trackname})


while True:
    state=sim.get_state()
    p=model.predict(project.State2X(state))
    steering=p[0][0,0]
    throttle=p[1][0,0]
    print("steering {:5.3f} throttle {:5.3f} pathdistance {:7f} offset {:5f} PID {:5.3f} {:5.3f} dt={:5.4f}".format(steering,throttle,state["pathdistance"], state["pathoffset"], state["PIDthrottle"], state["PIDsteering"],state["delta_time"]))
    sim.send_cmd({"steering":steering,'throttle':throttle})