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

input = h5py.File(filename, 'r')
imagesin=input['frontcamera']
controlsin=input['steering.throttle']

nsamples=imagesin.shape[0]
ntrain=int(nsamples*0.9)
nval=nsamples-ntrain


#imagesin = HDF5Matrix(filename, 'frontcamera',start=0,end=1000)
#controlsin = HDF5Matrix(filename, 'steering.throttle',start=0,end=1000)

nsamples,height,width,channels=imagesin.shape

print(imagesin.shape,controlsin.shape)

print(np.mean(imagesin[0:100]))
print(np.std(imagesin[0:100]))
print(np.mean(controlsin[0,0:100]))

#datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True)


def createCNNModel():
    # Create the model
    inp = Input((height,width, 3))
    model = Sequential()
    model.add(Conv2D(32,(3, 3), input_shape=(height,width, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3),  padding='same', activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3),  padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    x=Dense(32, activation='relu')(model(inp))
    steering_model=Dense(1, activation='linear',name="steering")(x)
    x=Dense(32, activation='relu')(model(inp))
    throttle_model=Dense(1, activation='linear',name="throttle")(x)
    combined_model=Model(inp,[steering_model,throttle_model])
    # Compile models
    opt = Adam(lr=0.00001)
    combined_model.compile(loss='mean_squared_error', optimizer=opt)
    return combined_model

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
model = createCNNModel()
print("CNN Model created.")
print(np.mean(imagesin[100:120]),np.std(imagesin[100:120]))
model.fit_generator(generator(imagesin[:ntrain],controlsin[:ntrain]), steps_per_epoch=100 ,verbose=1,
                    validation_data=generator(imagesin[ntrain:],controlsin[ntrain:]),validation_steps=10,
                    epochs=10,callbacks=[ModelCheckpoint("model_1e.h5")])
print("evaluate")
print(model.evaluate_generator(generator(imagesin[ntrain:],controlsin[ntrain:]), 1))
#print("Predict")
#print(model.predict_generator(generator(imagesin[ntrain:],controlsin[ntrain:]), 10))
model.save(model_filename)

