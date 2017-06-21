import h5py
import pickle
import numpy as np
import sys
from keras.models import load_model
import simulator
import random

# model names
model_filename="../data/model_rl_{}.h5"
filename_h5="../data/RL_experience_{}.h5"
experience_count=10000 # length of each recording

steering_sigma=0.005
throttle_sigma=0.02

# Reinforcement controller


policy_gen=1

def open_h5(generation,shape):
    output = h5py.File(filename_h5.format(generation), 'w')
    images = output.create_dataset('frontcamera',shape, 'i1')
    images.attrs['description'] = "simple test"
    controls = output.create_dataset('steering.throttle', (shape[0], 2))
    rewards = output.create_dataset('rewards', (shape[0], 3))
    return output,images,controls,rewards


def reward_func(distance, offset):
    if abs(offset > 100):
        return -100
    ret = distance - reward_func.last_distance
    reward_func.last_distance=distance
    if (ret<0):
        return 0
    return ret*0.01
reward_func.last_distance=0

def retrain(model,generation):

    input = h5py.File(filename_h5.format(generation), 'r')
    retrain_imagesin = input['frontcamera']
    retrain_controlsin = input['steering.throttle']
    retrain_reward=input['rewards']
    wl=10
    train_sz=retrain_reward.shape[0]-wl
    advantage=np.zeros_like(retrain_controlsin[:train_sz])
    discount_factor=0.9
    for n in range(0,train_sz):
        discount=1
        rv=0
        for m in range(0,wl):
            rv=retrain_reward[n+m]*discount
            discount *= discount_factor
        rv /= wl
        advantage[n] = rv/wl

        retrain_controlsin[n] * (1 + rv)

    advantage = (advantage - np.mean(advantage)) / np.std(advantage)  # improves training per karpathy

    advantaged_controls=retrain_controlsin[:train_sz]*(1+advantage)
    print(advantaged_controls[100])
    model.fit(retrain_imagesin[:train_sz], advantaged_controls, verbose=1, validation_split=0.2,epochs=10,shuffle="batch" )
    # we should test if validation results improved
    print (model.summary())
    return model

#model = load_model(model_filename.format(policy_gen))
#model=retrain(model,policy_gen)
#exit()

#connect to simulator
sim=simulator.Simulator()
sim.connect()

print("loading model {}".format(1))
model = load_model(model_filename.format(1))
for policy_gen in range(1,100):
    print (model.summary())
    dshape=model.input_shape
    print(dshape)
    dshape=[experience_count,90,160,3]
    output,images,controls,reward=open_h5(policy_gen,dshape)

    for h5idx in range(experience_count):
        state=sim.get_state()
        img=state["frontcamera"]
        img=np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
        predict=model.predict(img)
        steering = random.gauss(predict[0,0],steering_sigma)
        throttle = random.gauss(predict[0,1],throttle_sigma)
        rwrd=reward_func(state["pathdistance"],state["offset"])
        print("steering {:5.3f} throttle= {:5.3f} reward={:4.2f} pathdistance {:7f} offset {:5f} PID {:5.3f} {:5.3f} dt={:5.4f}"
              .format(steering,throttle,rwrd,state["pathdistance"], state["offset"], state["PIDthrottle"], state["PIDsteering"],state["delta_time"]))
        sim.send_cmd({"steering":steering,'throttle':state["PIDthrottle"]})

        #record experience
        controls[h5idx] = [steering,throttle] #record the deviated controls
        images[h5idx] = state["frontcamera"]
        reward[h5idx] =[rwrd,steering_bias,throttle_bias]

    output.flush()
    output.close()

    #update the model
    model=retrain(model,policy_gen)
    print("save model {}".format(policy_gen+1))
    model.save(model_filename.format(policy_gen+1))
