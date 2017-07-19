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
from h5utils import h5shuffle

def train(project, model_filename, filename="robocar.hdf5",train_filename="train.hdf5"):
    # use project directories
    pathname=os.path.join(project.datadir,filename )
    train_pathname=os.path.join(project.datadir,train_filename)
    checkpoint_pathname=os.path.join(project.modeldir,"checkpoint_"+model_filename)
    model_pathname=os.path.join(project.modeldir,"model_1.h5")

    #shuffle the data file if needed
    if os.path.exists(train_pathname) or not os.path.getmtime(train_pathname) > os.path.getmtime(pathname):
        h5shuffle(pathname, train_pathname)

    #open input file
    input = h5py.File(train_pathname, 'r')
    recorder=project.recorder(input)
    actions=input['actions']
    config=recorder.config
    nsamples=recorder.nsamples

    # create our CNN model
    model = project.createModel(config)
    ninputs=len(model.input_shape)
    print("Model created.",ninputs)
    model.summary()

    #imitation learning from recorded actions
    def gety(idx,cnt):
        return project.converty(input['actions'][idx:idx+cnt])

    model.fit_generator(recorder.generator(range(int(nsamples*0.8)),gety),
            steps_per_epoch=4,
            epochs=10,
            verbose=1,
            #shuffle="batch",
            #validation_data=recorder.generator(range(int(nsamples*0.8)),actions=True),
            #validation_steps=100,
            callbacks=[ModelCheckpoint(checkpoint_pathname)])

    print("evaluate")
    print(model.metrics_names)
    print(model.evaluate_generator(recorder.generator(range(int(nsamples*0.8)),gety),steps=10))
    model.save(model_pathname)

if __name__ == "__main__":
    # connect to environment
    train(project,project.model_filename)