import h5py
import pickle
import numpy as np
import sys
from keras.models import load_model
import arobocar
import random
import os
import project
import importlib
import gym

# Reinforcement controller
# experience_count - length of each recording
# sigma - noise to add to RL actions [steering,throttle]
# first_gen - starting with generation - will load saved model (first_gen-1)

def run(project,env,
        initial_model_filename,
        sigma,
        experience_count=10000,
        first_gen=1,
        model_filename="model_rl_{}.h5",
        experience_filename="RL_experience_{}.h5",
        ):
    # use project directories
    model_pathname=os.path.join(project.modeldir,model_filename) # model names
    experience_pathname=os.path.join(project.datadir,experience_filename) # temporary experience file

    #select initial model
    if initial_model_filename:
        initial_model_pathname = os.path.join(project.modeldir, initial_model_filename)  # model names
        model = load_model(initial_model_pathname)
    else:
        print("loading pretrained model {}".format(first_gen))
        model = load_model(model_pathname.format(first_gen))

    print(model.summary())
    print(model.input_shape)

    for policy_gen in range(first_gen,100):
        # set up experience recording
        output = h5py.File(experience_pathname.format(policy_gen), 'w')
        recorder = project.recorder(output, project.config, experience_count)
        actions = output.create_dataset('actions', (experience_count, 2))
        sample_noise = output.create_dataset('sample_noise', (experience_count, 2))

        reward_sum=0
        resetidx=0
        reset_cnt=0
        reset_reward_sum=0

        observation = env.reset()
        info = None
        for h5idx in range(experience_count):
            action = project.converta(model.predict(project.convertX(observation, info)))
            noise=[random.gauss(0,s) for s in sigma]
            action=np.add(action,noise)

            # get the next observation
            observation, reward, done, info = env.step(action)
            sensor=observation[1]
            if (done):
                print("Done reward={} odometer={}".format(reward_sum,observation[1][2]))
                env.reset()
                resetidx=h5idx
                reset_reward_sum=reward_sum
                reset_cnt+=1
                continue

            reward_sum += reward

            print("{:6}/{:6} tsr {:6} reward {:10.0f} steering {:+5.3f} throttle= {:+5.3f}{:+5.3f} reward={:4.2f} speed {:+5.3f} "
                  .format(h5idx,experience_count,h5idx-resetidx,reward_sum-reset_reward_sum,0,1,noise[1],reward,0))
            print(sensor)

            #record experience
            idx=recorder.record(observation,reward,done,info)
            actions[idx] =  action #record the undeviate controls
            sample_noise[idx] = noise # and deviations separately

        print("Generation {} avg reward={} resets={}".format(policy_gen,reward_sum/experience_count,reset_cnt))
        print("mean {}".format(np.mean(recorder.rewards)))

        output.flush()
        output.close()
        env.reset()

        #update the model
        print("Retrain Generation {}".format(policy_gen))
        input = h5py.File(experience_pathname.format(policy_gen), 'r')
        model=project.retrain(input,model)
        print("save model {}".format(policy_gen+1))
        model.save(model_filename.format(policy_gen+1))

if __name__ == "__main__":
    # connect to environment
    env = gym.make('ARoboCar-v0')
    experience_count=10000
    sigma=[0.005,0.0002]
    run( project, env, project.model_filename,sigma)
