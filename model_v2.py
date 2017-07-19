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

# convert observation+info structure and normalize for model
def preprocess(observation,info):
    ret=observation
    ret[1] *= np.array([0.001,0.001,0.00001])
    return ret

#convert output of model to structure/units used by arobocar
def converta(actions):
    return actions

#convert observation+info to input to model
def convertX(observation,info):
    img = observation[0]
    img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    sensor = observation[1]
    sensor = np.reshape(np.array([sensor[0] * .001, sensor[1] * .001, 0.0]), (1, 3))
    return [img,sensor]

scaled=np.array([0.001,0.001,0.00001])
#record any additional items
class recorder(H5Recorder):
    def __init__(self,h5file,config=None,maxidx=10000):
        super().__init__(h5file, config,maxidx,preprocess=preprocess,postprocess=postprocess)
        if config: # only set for create
            self.ec=h5file.create_dataset('ec', (maxidx, 3), 'f')
            self.ec.attrs['cols']="speed,acceleration,odometer"
        else:
            self.ec=h5file['ec']

    # record a single observation
    def record(self, observation, reward, done, info):
        idx=super().record(observation,reward,done,info)
        self.ec[idx]=info['ec']
        return idx

    # get an array of observations
    def get(self, idx, cnt=1):
        observation, reward, done, info = super().get(idx,cnt)
        for i in range(cnt):
            info[i]['ec']=self.ec[idx+i]
        return observation, reward, done, info


    # returns an array of observations,reward,..
    def getX(self,idx,cnt=1):
        X=[self.images[idx:idx+cnt],self.sensors[idx:idx+cnt]*scaled]
        return X



