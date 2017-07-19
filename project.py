from model_v2 import *
from arobocar import ARoboCar
import gym
datadir="../data/"
modeldir="../data/"
from PGRL import *

model_filename="model_1.h5"

config = {"camerawidth": 128,
          "cameraheight": 160,
          "trackname": "Racetrack1",
          "cameraloc": [50, 0, 200],
          "camerarot": [0, -30, 0],
          "observer": 'EmbeddedObserver',
}

gym.envs.register(id='ARoboCar-v0',
                  entry_point='arobocar:ARoboCar',
                  kwargs={'config':config}
)
