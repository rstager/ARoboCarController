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

filename=os.path.join(project.datadir,"robocar.hdf5")
model_filename=os.path.join(project.modeldir,"model_1.h5")
checkpoint_filename=os.path.join(project.modeldir,"model_1.h5")

input = h5py.File(filename, 'r')
config, nsamples, datasets=project.getDatasets(input)
controlsin=input['steering.throttle']

ntrain=int(nsamples*0.9)
nval=nsamples-ntrain


#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True)




def generator(Xh5,yh5):
    m=Xh5.shape[0]
    s=0
    while 1:
        e=s+32
        X=Xh5[s:e]
        y=yh5[s:e]
        y=[yh5[s:e,0],yh5[s:e,1]]
        yield (X,y)
        s +=32
        if s+32 > m:
            s = 0

# create our CNN model
model = project.createModel(config)
print("Model created.")
model.fit(datasets, [controlsin[:,0],controlsin[:,1]], verbose=1,
                    validation_split=0.2,
                    epochs=10,callbacks=[ModelCheckpoint(checkpoint_filename)])
print("evaluate")
print(model.evaluate(datasets,[controlsin[:,0],controlsin[:,1]]))
#print("Predict")
#print(model.predict_generator(generator(imagesin[ntrain:],controlsin[ntrain:]), 10))
model.save(model_filename)

