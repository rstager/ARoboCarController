import keras
from keras.layers import Conv2D,Dropout,MaxPooling2D,Dense,Flatten,BatchNormalization
from keras.models import Sequential,Model,Input
from keras.losses import mean_absolute_error,mean_squared_error
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.utils.io_utils import HDF5Matrix
from keras.preprocessing import *
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import project
import random
from utils import h5shuffle

filename=os.path.join(project.datadir,"robocar.hdf5")
model_filename=os.path.join(project.modeldir,"model_1.h5")
checkpoint_filename=os.path.join(project.modeldir,"model_1.h5")
train_filename=os.path.join(project.datadir,"train.hdf5")
if not os.path.exists(train_filename) or not os.path.getmtime(train_filename) > os.path.getmtime(filename):
    h5shuffle(filename,train_filename)
input = h5py.File(train_filename, 'r')
config, nsamples, datasets =project.getDatasets(input)
controlsin=input['steering.throttle']

# create our CNN model
model = project.createModel(config)
ninputs=len(model.input_shape)
print("Model created.",ninputs)
model.summary()

model.fit([datasets[0][:nsamples],datasets[1][:nsamples]],
          [controlsin[:nsamples, 0].reshape(nsamples, 1), controlsin[:nsamples, 1].reshape(nsamples, 1)], verbose=1,
          validation_split=0.1,
          shuffle="batch",
          epochs=10,
          callbacks=[ModelCheckpoint(checkpoint_filename)])

print("evaluate")
print(model.metrics_names)
print(model.evaluate(datasets[:ninputs],[controlsin[:,0],controlsin[:,1]]))
model.save(model_filename)

