import h5py
import pickle
import numpy as np
import sys
from keras.models import load_model
import simulator
import random
import os
import project
import importlib


# Reinforcement controller



model_filename=os.path.join(project.modeldir,"model_rl_{}.h5") # model names
filename_h5=os.path.join(project.datadir,"RL_experience_{}.h5") # temporary experience file
first_gen=1 # starting with generation - will load saved model (first_gen-1)
experience_count=10000 # length of each recording
sigma=[0.005,0.0002] # noise to add to RL controls [steering,throttle]
imitation_count=0 # first N generations w
# ill mimic PID controller, set initial_model_name if this is zero
initial_model_filename=os.path.join(project.modeldir,"model_racetrack2_fulltraining.h5")

def open_h5(generation,nsamples,camerashape):
    output = h5py.File(filename_h5.format(generation), 'w')
    datasets = project.createDatasets(config, output, nsamples)
    controls = output.create_dataset('steering.throttle', (nsamples, 2))
    rewards = output.create_dataset('rewards', (nsamples,1))
    noise = output.create_dataset('sample_noise', (nsamples, 2))
    return output,datasets,controls,rewards,noise


def reward_func(state):
    speed=state["speed"]
    offset=state["pathoffset"]
    if abs(offset) > 100:
        return -1000-speed
    #return speed*0.01-abs(state['pathoffset'])*.01
    return speed * 0.01
reward_func.last_distance=0



#connect to simulator
sim=simulator.Simulator()
config=sim.connect({"trackname":project.trackname,'controller':importlib.util.find_spec("EmbeddedController").origin})


steering_noise=.15      #amount of noise to add to steering
noise_probability=0.003  #how often to deviate - set to zero to drive correctly
deviation_duration=40   # duration of deviation
deviating_cnt=0
deviation_angle=0

def imitation_predict(state):
    global deviating_cnt,deviation_angle,deviation_duration
    steering=state["PIDsteering"]
    throttle=state["PIDthrottle"]
    offset=state["pathoffset"] #distance from center of road
    noise=0

    if deviating_cnt > 0 and abs(offset) > 75:  # stop deviating if we ran off the road
        deviating_cnt = 0
        print("Abort deviation")

    if deviating_cnt>0: # while deviation
        noise=deviation_angle - steering #add noise to steer deviation_angle
        deviating_cnt -= 1
        if (deviating_cnt == 0):
            print("End deviation")

    #decide when to start another deviation
    if deviating_cnt == 0 and random.random() < noise_probability:
        deviating_cnt = deviation_duration
        deviation_angle = steering + random.random() * steering_noise - (steering_noise / 2)
        print("** Begin Steering deviation {}".format(deviation_angle))

    return [steering, throttle],[noise,0]
if initial_model_filename:
    model = load_model(initial_model_filename)
elif imitation_count >= first_gen :
    print("create new model ".format())
    model = project.createModel(config)
else:
    print("loading pretrained model {}".format(first_gen))
    model = load_model(model_filename.format(first_gen))
print(model.summary())
ninputs=len(model.input_shape)
print(model.input_shape)

offroad_cnt=0
height = config["cameraheight"]
width = config["camerawidth"]

for policy_gen in range(first_gen,100):
    reward_sum=0
    dshape=model.input_shape
    print("Input Shape {}".format(dshape))
    output,datasets,controls,reward,sample_noise=open_h5(policy_gen,experience_count,(height,width,3)) #todo:change to take config

    imitating = policy_gen <= imitation_count
    resetidx=0
    reset_cnt=0
    reset_reward_sum=0
    for h5idx in range(experience_count):
        state=sim.get_state()
        Xs=project.State2X(state)
        if imitating:
            predict,noise=imitation_predict(state)
        else:
            p = model.predict(Xs[:ninputs])
            predict=[p[0][0, 0],p[1][0, 0]]
            noise=[random.gauss(0,s) for s in sigma]
        control=np.add(predict,noise)
        rwrd=reward_func(state) # todo: move to simulator.py
        #print("steering {:+5.3f}{:+5.3f} throttle= {:+5.3f}{:+5.3f} reward={:4.2f} pathdistance {:10.1f} offset {:+5.1f} PID {:+5.3f} {:+5.3f} dt={:5.4f}"
        #      .format(predict[0],noise[0],predict[1],noise[1],rwrd,state["pathdistance"], state["pathoffset"], state["PIDthrottle"], state["PIDsteering"],state["delta_time"]))
        print("{:6}/{:6} tsr {:6} reward {:10.0f} steering {:+5.3f} throttle= {:+5.3f}{:+5.3f} reward={:4.2f} speed {:+5.3f} "
              .format(h5idx,experience_count,h5idx-resetidx,reward_sum-reset_reward_sum,predict[0],predict[1],noise[1],rwrd,state["speed"]))
        sim.send_cmd({"steering":control[0],'throttle':control[1]})

        #record experience
        controls[h5idx] =  predict #record the undeviate controls
        sample_noise[h5idx] = noise # and deviations separately
        for ds, x in zip(datasets, Xs):  ds[h5idx] = x
        reward[h5idx] =[rwrd]

        if(h5idx==experience_count-1):
            print("final distance={}".format(state["pathdistance"]))
        if (state["pathoffset"]>200):
            offroad_cnt +=1
            if(offroad_cnt>10):
                print("Offroad resetting")
                sim.reset()
                offroad_cnt=0
                resetidx=h5idx
                reset_reward_sum=reward_sum
                reset_cnt+=1
        else:
            offroad_cnt =0
        reward_sum += rwrd

    print("Generation {} avg reward={} resets={}".format(policy_gen,reward_sum/experience_count,reset_cnt))
    print("mean {}".format(np.mean(reward)))

    output.flush()
    output.close()
    sim.reset()

    #update the model
    print("Retrain Generation {}".format(policy_gen))
    input = h5py.File(filename_h5.format(policy_gen), 'r')
    model=project.retrain(input,model)
    print("save model {}".format(policy_gen+1))
    model.save(model_filename.format(policy_gen+1))
