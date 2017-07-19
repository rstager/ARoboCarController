# display a frame saved from AIAgent.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import project

input = h5py.File(os.path.join(project.datadir,"robocar.hdf5"), 'r')
recorder=project.recorder(input)

print("Number of samples={}".format(recorder.nsamples))
fig, ax = plt.subplots()

first=True
for idx in range(recorder.nsamples):
    action=input['actions'][idx]
    img=input['frontcamera'][idx]
    sensor=input['sensors'][idx]
    reward=input['rewards'][idx]

    print("{} sensors {} actions {} reward {}".format(idx,sensor,action,reward,img.shape))
    if first:
        im = ax.imshow(img*255)
        fig.show()
        first=False
    else:
        im.set_data(img*255)
    fig.canvas.draw()
