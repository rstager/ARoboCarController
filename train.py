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
from h5utils import h5shuffle,h5gety

from functools import partial

def train(project, model, gety, filename="robocar.hdf5",train_filename="train.hdf5",checkpoint_filename="checkpoint"):
    # use project directories
    pathname=os.path.join(project.datadir,filename )
    train_pathname=os.path.join(project.datadir,train_filename)
    checkpoint_pathname=os.path.join(project.modeldir,checkpoint_filename)

    #shuffle the data file if needed
    if not os.path.exists(train_pathname) or not os.path.getmtime(train_pathname) > os.path.getmtime(pathname):
        h5shuffle(pathname, train_pathname)

    #open input file
    input=h5py.File(train_pathname, 'r')
    recorder=project.recorder(input)
    nsamples=recorder.nsamples
    gety2=partial(gety,input)


    model.fit_generator(recorder.generator(range(int(nsamples*0.8)),gety2),
            steps_per_epoch=4,
            epochs=10,
            verbose=1,
            #shuffle="batch",
            #validation_data=recorder.generator(range(int(nsamples*0.8)),actions=True),
            #validation_steps=100,
            callbacks=[ModelCheckpoint(checkpoint_pathname)])

def evaluate(project,model,gety,filename,num=10):
    #open input file
    pathname=os.path.join(project.datadir,filename )
    input=h5py.File(pathname, 'r')
    recorder=project.recorder(input)
    gety2=partial(gety,input)
    print("evaluate")
    print(model.metrics_names)
    print(model.evaluate_generator(recorder.generator(None,gety2),steps=num))

    #imitation learning from recorded action

def run (project, gety=None, model_filename=None,  filename = "robocar.hdf5"):
    if not model_filename: model_filename=project.model_filename
    if not gety: gety=h5gety
    model = project.createModel(project.config)
    model.summary()
    gety3=partial(gety, project, 'actions')
    train(project,model,gety3)
    model_pathname=os.path.join(project.modeldir,model_filename)
    model.save(model_pathname)
    evaluate(project,model,gety3,filename=filename)

if __name__ == "__main__":
    # connect to environment
    run(project)