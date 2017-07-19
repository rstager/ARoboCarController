import h5py
import pickle
import numpy as np
import sys
from keras.models import load_model
import arobocar
import os
import project
import importlib
import gym
from utils import listexpand

def run(project,env,model_filename):
    model=load_model(os.path.join(project.modeldir,"model_1.h5"))
    print(model.summary())
    ninputs=len(model.input_shape)
    observation=env.reset()
    info=None

    while True:
        action=project.converta(model.predict(project.convertX(observation,info)))
        observation,reward,done,info = env.step(action)
        print("action {} reward={:5.4f}".format(action,reward))
        if(done):
            print("Done")
            env.reset()

if __name__ == "__main__":
    # connect to environment
    env = gym.make('ARoboCar-v0')
    run( project, env, project.model_filename)