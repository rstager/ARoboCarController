from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.layers.merge import Concatenate,concatenate
from keras.models import Sequential, Model, Input
from keras.optimizers import Adam
import numpy as np
import pickle


def createModel(config):
    height=config["cameraheight"]
    width = config["camerawidth"]
    # Create the model
    inp = Input((height,width, 3),name="frontcamerainput")
    sensors = Input((1,),name="speedinput") #just speed for now
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
    x=concatenate([model(inp),sensors])
    x=Dense(32, activation='relu')(x)
    steering_model=Dense(1, activation='linear',name="steering")(x)
    x=Dense(32, activation='relu')(model(inp))
    throttle_model=Dense(1, activation='linear',name="throttle")(x)
    combined_model=Model([inp,sensors],[steering_model,throttle_model])
    # Compile models
    opt = Adam(lr=0.00001)
    combined_model.compile(loss='mean_squared_error', optimizer=opt)
    return combined_model

#defines the shape of the model input vectors for creation
def Xshape(config):
    height=config["cameraheight"]
    width = config["camerawidth"]
    return {"frontcamera":(height, width, 3), "speed":(1,)}

def State2X(state):
    img = state["frontcamera"]
    speed = np.array([state["speed"]])*0.001
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    return [img,speed]


def createDatasets(config,output,maxidx):
    height=config["cameraheight"]
    width = config["camerawidth"]
    images = output.create_dataset('frontcamera', (maxidx, height, width, 3), 'i1')
    images.attrs['description'] = "simple test"
    speeds = output.create_dataset('speed', (maxidx, 1), 'f')
    output.attrs["config"]=pickle.dumps(config,0)
    return [images,speeds]

def getDatasets(input):
    config= pickle.loads(input.attrs["config"])
    nsamples = input['frontcamera'].shape[0]
    return config, nsamples, [input['frontcamera'],input['speed']]