import keras
import numpy as np
import os, sys, math
from config.constants import *

def get_batch(splitName,
    chunkPath,
    isShuffle=True,
    chunkCount=0):

    '''
    Python data generator to supply data in batches to keras trainer.

    Parameters: 
        chunkPath - path to the numpy files containing defect patches
        isShuffle - boolean value for shuffle np chunks for each epoch
        chunkCount - number of chunks to be used
    '''

    if os.path.isdir(data_path) is not True:
        # print('current dir is {}'.format(os.getcwd()))
        print("Data directory \"{}\" does not exist".format(data_path))
        sys.exit()

    indexes = np.arange(chunkCount, dtype=np.uint8)

    #Serve the data in infinite loop. This generator is supposed to be 
    #used from fit, predict and evaluate generators in keras

    while(1):
        if isShuffle is True:
            np.random.shuffle(indexes)

        for i in indexes:
            chunk_data_path = os.path.join(data_path, 
            chunkPath.format(splitName, i, 'x'))
            chunk_label_path = os.path.join(data_path, 
            chunkPath.format(splitName, i, 'y'))
        
            data = np.load(chunk_data_path)
            label = np.load(chunk_label_path)

            label = keras.utils.to_categorical(label, 
                                            num_classes)

            #Channels last as required by tensorflow backend
            data = np.transpose(data,[0,2,3,1])

            num_of_minibatches = math.ceil(data.shape[0] / batch_size)
            for k in range(num_of_minibatches):
                yield data[k*batch_size: (k+1) * batch_size], label[k*batch_size: (k+1) * batch_size]

            #Yield the remaining minibatches
            if(len(data[num_of_minibatches*batch_size:]) > 0):
                yield data[num_of_minibatches*batch_size:], label[num_of_minibatches*batch_size]