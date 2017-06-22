import h5py
import pickle
import numpy as np
import sys
from keras.models import load_model
import simulator
import random
import os
import project

# model names
initial_model_filename=os.path.join(project.modeldir,"model_1.h5")
model_filename=os.path.join(project.modeldir,"model_rl_{}.h5")
filename_h5=os.path.join(project.datadir,"RL_experience.h5")
experience_count=10000 # length of each recording
sigma=[0.005,0.02]


# Reinforcement controller


policy_gen=1

def open_h5(generation,shape):
    output = h5py.File(filename_h5.format(generation), 'w')
    images = output.create_dataset('frontcamera',shape, 'i1')
    images.attrs['description'] = "simple test"
    controls = output.create_dataset('steering.throttle', (shape[0], 2))
    rewards = output.create_dataset('rewards', (shape[0],1))
    noise = output.create_dataset('sample_noise', (shape[0], 2))
    return output,images,controls,rewards,noise


def reward_func(state):
    speed=state["speed"]
    offset=state["pathoffset"]
    if abs(offset) > 100:
        return -1000-speed
    return speed*0.01-abs(state['pathoffset'])*.01
reward_func.last_distance=0

def retrain(model,generation):
    input = h5py.File(filename_h5.format(generation), 'r')
    retrain_imagesin = input['frontcamera']
    retrain_controlsin = input['steering.throttle'] # original policy without deviation
    retrain_reward=input['rewards']
    retrain_noise=input["sample_noise"]             # deviation
    nsamples=retrain_imagesin.shape[0]
    wl=60 # how far to calculate discounted future reward
    advantage=np.zeros_like(retrain_controlsin)
    discount_factor=0.97
    for n in range(0,nsamples-1):
        discount=1
        rv=0
        twl=min(wl,nsamples-n)
        for m in range(0,twl-1):
            rv=retrain_reward[n+m]*discount
            discount *= discount_factor
        rv /= twl
        advantage[n] = rv/twl
    advantage = (advantage - np.mean(advantage)) / np.std(advantage)  # improves training per karpathy
    advantaged_controls=retrain_controlsin + retrain_noise*advantage
    advantaged_controls[:,1]=0.7
    print("Generation {}".format(generation))
    print("advantage={} noise={} product={}".format(np.mean(advantage,axis=0),np.mean(retrain_noise,axis=0),np.mean(retrain_noise*advantage,axis=0)))
    print("avg orig steer={} throttle={}".format(np.mean(retrain_controlsin[:, 0]), np.mean(retrain_controlsin[:, 1])))
    print("avg adv  steer={} throttle={}".format(np.mean(advantaged_controls[:, 0]), np.mean(advantaged_controls[:, 1])))
    #print("Learning Rate:{}".format(model.qlr.get_value()))
    model.lr=0.0001
    model.fit(retrain_imagesin, [advantaged_controls[:,0],advantaged_controls[:,1]], verbose=2, validation_split=0.2,batch_size=100,epochs=3,shuffle="batch")
    # we should test if validation results improved
    return model

#connect to simulator
sim=simulator.Simulator()
sim.connect()

print("loading model {}".format(1))
model = load_model(initial_model_filename.format(1))
offroad_cnt=0
for policy_gen in range(1,100):
    dshape=model.input_shape
    dshape=[experience_count,90,160,3]
    output,images,controls,reward,sample_noise=open_h5(policy_gen,dshape)

    for h5idx in range(experience_count):
        state=sim.get_state()
        img=state["frontcamera"]
        img=np.reshape(img,(1,img.shape[0],img.shape[1],img.shape[2]))
        #predict=model.predict(img)[0]
        p = model.predict(img)
        predict=[p[0][0, 0],p[1][0, 0]]
        noise=[random.gauss(0,s) for s in sigma]
        control=predict+noise
        rwrd=reward_func(state)
        print("steering {:+5.3f}{:+5.3f} throttle= {:+5.3f}{:+5.3f} reward={:4.2f} pathdistance {:10.1f} offset {:+5.1f} PID {:+5.3f} {:+5.3f} dt={:5.4f}"
              .format(predict[0],noise[0],predict[1],noise[1],rwrd,state["pathdistance"], state["pathoffset"], state["PIDthrottle"], state["PIDsteering"],state["delta_time"]))
        sim.send_cmd({"steering":control[0],'throttle':control[1]})

        #record experience
        controls[h5idx] =  predict #record the undeviate controls
        sample_noise[h5idx] = noise # and deviations separately
        images[h5idx] = state["frontcamera"]
        reward[h5idx] =[rwrd]
        if(h5idx==experience_count-1):
            print("final distance={}".format(state["pathdistance"]))
        if (state["pathoffset"]>200):
            offroad_cnt +=1
            if(offroad_cnt>100):
                print("Offroad resetting")
                sim.reset()
                offroad_cnt=0
        else:
            offroad_cnt =0
    output.flush()
    output.close()

    sim.reset()

    #update the model
    print("Retrain")
    model=retrain(model,policy_gen)
    print("save model {}".format(policy_gen+1))
    model.save(model_filename.format(policy_gen+1))
