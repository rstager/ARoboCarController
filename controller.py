import h5py
import pickle
import numpy as np
import sys
import simulator
import project

# This controller just follows the PID recommendations
sim=simulator.Simulator()
config=sim.connect(project.connection_properties)
print("Config=",config)


while True:
    state=sim.get_state()
    print("pathdistance {:7.0f} offset {:5.0f} throttle {:1.3f} speed {:1.3f} angle {:1.3f} dt={:5.4f}".format(state["pathdistance"], state["pathoffset"], state["PIDthrottle"], state["PIDspeed"],state["PIDsteering"],state["delta_time"]))
    sim.send_cmd({"steering":state["PIDsteering"],'throttle':state["PIDthrottle"]})
