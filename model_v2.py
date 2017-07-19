from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.layers.merge import Concatenate,concatenate
from keras.models import Sequential, Model, Input
from keras.optimizers import Adam
import numpy as np
import pickle
from h5utils import H5Recorder


def createModel(config):
    height=config["cameraheight"]
    width = config["camerawidth"]
    # Create the model
    inp = Input((height,width, 3),name="frontcamerainput")
    sensors = Input((3,),name="sensors") # speed, accelleration, throttle
    model = Sequential()
    model.add(Conv2D(32,(3, 3), input_shape=(height,width, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32,(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3),  padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3),  padding='same', activation='relu'))
    model.add(Conv2D(64,(3, 3),  padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    fullx=concatenate([model(inp),sensors])
    x=Dense(32, activation='relu')(fullx)
    steering_model=Dense(1, activation='linear',name="steering")(x)
    x=Dense(4, activation='relu')(fullx)
    throttle_model=Dense(1, activation='linear',name="throttle")(x)
    combined_model=Model([inp,sensors],[steering_model,throttle_model])
    # Compile models
    opt = Adam(lr=0.00001)
    combined_model.compile(loss='mean_squared_error', optimizer=opt)
    return combined_model


scaled=np.array([0.001,0.001,0.00001])

#convert structure and normalize single observation+info to input to model
def convertX(observation,info):
    X=[np.expand_dims(np.array(x),axis=0) for x in observation]
    X[1]*=scaled
    return X

#convert output of model to structure/units used by arobocar: single action
def converta(y):
    return [y[0][0],y[1][0]]

#convert array of arobocar actions to model y: used for imitation learning
def converty(action):
    return [action[:,0],action[:,1]]

#record any additional items
class recorder(H5Recorder):
    def __init__(self,h5file,config=None,maxidx=10000):
        super().__init__(h5file, config,maxidx)
        if config: # only set for create
            self.ec=h5file.create_dataset('ec', (maxidx, 3), 'f')
            self.ec.attrs['cols']="speed,acceleration,odometer"

    # record a single observation
    def record(self, observation, reward, done, info):
        idx=super().record(observation,reward,done,info)
        self.ec[idx]=info['ec']
        return idx

    # get batch of X from h5file, change structure and scale for model
    def getX(self,idx,cnt=1):
        X=[self.images[idx:idx+cnt],self.sensors[idx:idx+cnt]*scaled]
        return X




