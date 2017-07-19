import h5py
from os import unlink,rename
from tempfile import mktemp
from random import shuffle
import pickle
import numpy as np
import threading

def h5shuffle(input_filename, output_filename=None):
    tmp_filename=mktemp()
    if not output_filename:
        output_filename=input_filename
    input = h5py.File(input_filename, 'r')
    output = h5py.File(tmp_filename, 'w')
    if hasattr(input, 'attrs'):
        for k, v in input.attrs.items(): output.attrs[k] = v
    shuffle_idxs=None
    for name,dataset in input.items():
        print("copy {} {}".format(name,dataset.shape))
        output_dataset= output.create_dataset(name, dataset.shape, dataset.dtype)
        if hasattr(dataset, 'attrs'):
            for k, v in dataset.attrs.items(): output_dataset.attrs[k] = v
        if not shuffle_idxs:
            shuffle_idxs=[x for x in range(dataset.shape[0])]
            shuffle(shuffle_idxs)
        for idx,value in enumerate(shuffle_idxs):
            output_dataset[idx]=dataset[value]
    input.close()
    output.close()
    if output_filename==input_filename:
        unlink(input_filename)
    rename(tmp_filename,output_filename)

class H5Recorder:
    def __init__(self,h5file,config=None,maxidx=None,preprocess=lambda o,i: o,postprocess=lambda a: a):
        self.h5file=h5file
        self.h5idx = -1
        self.preprocess=preprocess
        self.postprocess=postprocess
        if config!=None:
            self.config=config
            height = config["cameraheight"]
            width = config["camerawidth"]
            self.images = self.h5file.create_dataset('frontcamera', (maxidx, height, width, 3), 'i1')
            self.images.attrs['description'] = "simple test"
            self.sensors = self.h5file.create_dataset('sensors', (maxidx, 3), 'f')
            self.sensors.attrs['cols'] = "speed,acceleration,odometer"
            self.rewards = h5file.create_dataset('rewards', (maxidx, 1))
            self.dones = h5file.create_dataset('dones', (maxidx, 1))
            self.h5file.attrs["config"] = pickle.dumps(config, 0)
        else:
            self.config = pickle.loads(h5file.attrs["config"])
            self.images= h5file['frontcamera']
            self.sensors= h5file['sensors']
            self.rewards= h5file['rewards']
            self.dones= h5file['dones']
        self.nsamples = self.rewards.shape[0]
        self.lock = threading.Lock()


    def record(self,observation,reward,done,info):
        self.h5idx += 1
        self.images[self.h5idx] = observation[0]
        self.sensors[self.h5idx] = observation[1]
        self.dones[self.h5idx] = done
        self.rewards[self.h5idx] = reward
        return self.h5idx

    # returns an array of observations,reward,..
    def get(self,idx,cnt=1):
        observation=[self.images[idx:idx+cnt],self.sensors[idx:idx+cnt]]
        done=self.dones[idx:idx+cnt]
        reward=self.rewards[idx:idx+cnt]
        info = [{}]*cnt
        return observation,reward,done,info

    # generate input and actions for training
    # training dataset can be ignored for predictions
    def generator(self, iter=None, actions=None, batchsz=32):
        with self.lock:
            if not iter:
                m=self.nsamples
                iter=range(m)
            while True:
                for idx in iter:
                    if actions:
                        action = actions[idx:idx+batchsz]
                        yield self.getX(idx,batchsz), self.converta(action)  # observations, actions
                    else:
                        yield self.getX(idx,batchsz)