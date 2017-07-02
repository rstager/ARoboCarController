from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.layers.merge import Concatenate,concatenate
from keras.models import Sequential, Model, Input
from keras.optimizers import Adam
import pickle
import numpy as np

#defines the shape of the model input vectors
def Xshape(config):
    height=config["cameraheight"]
    width = config["camerawidth"]
    return {"frontcamera":(height, width, 3)}

def createModel(config):
    height=config["cameraheight"]
    width = config["camerawidth"]
    global Xshape
    Xshape={"frontcamera":(height,width,3)}
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

def State2X(state):
    img = state["frontcamera"]
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    return [img]

def createDatasets(config,output,maxidx):
    height=config["cameraheight"]
    width = config["camerawidth"]
    images = output.create_dataset('frontcamera', (maxidx, height, width, 3), 'i1')
    images.attrs['description'] = "simple test"
    output.attrs["config"]=pickle.dumps(config,0)
    return [images]

def getDatasets(input):
    config= pickle.loads(input.attrs["config"])
    nsamples = input['frontcamera'].shape[0]
    return config, nsamples, [input['frontcamera']]


