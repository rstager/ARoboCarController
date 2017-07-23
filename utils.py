import numpy as np

# add first index for a list of np arrays
def listexpand(list):
    for a in list:
        np.expand_dims(a,axis=0)
    return list
