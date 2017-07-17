import h5py
from os import unlink,rename
from tempfile import mktemp
from random import shuffle

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