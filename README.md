# ARoboCarController
Controller for ARoboCar simulator
## Setup notes
Please set OS environment variable called DATA_DIR to a directory where the controller will write the training and model hdf5 files. It needs to be fully qualified and not a relative path as h5py isn't able to deal with relative paths on Mac. 
