# display a frame saved from AIAgent.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import project

input = h5py.File(os.path.join(project.datadir,"robocar.hdf5"), 'r')
config,nsamples,datasets=project.getDatasets(input)
imagesin=datasets[0]
anglesin=input['steering.throttle']

fig, ax = plt.subplots()

im = ax.imshow(imagesin[0])
fig.show()
for idx in range(imagesin.shape[0]):
    print(idx,anglesin[idx],imagesin[idx].shape)
    im.set_data(imagesin[idx]*255)
    fig.canvas.draw()
