# simple policy gradient RL
import numpy as np
import project


def retrain(input, model):
    config,nsamples,retrain_datasets = project.getDatasets(input)
    retrain_controlsin = input['steering.throttle'] # original policy without deviation
    retrain_reward=input['rewards']
    retrain_noise=input["sample_noise"]             # deviation

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
    #advantaged_controls[:,1]=0.7

    print("advantage={} noise={} product={}".format(np.mean(advantage, axis=0), np.mean(retrain_noise, axis=0),
                                                    np.mean(retrain_noise * advantage, axis=0)))
    retrain_y=[advantaged_controls[:,0],advantaged_controls[:,1]]
    model.lr=0.0001

    print("learning rate={}".format(model.lr))
    ninputs=len(model.input_shape)
    model.fit(retrain_datasets, retrain_y[:ninputs], verbose=2, validation_split=0.2,batch_size=100,epochs=3,shuffle="batch")

    # we should test if validation results improved
    print("Predict")
    p=model.predict(retrain_datasets, 20)
    print(p[:10])

    return model