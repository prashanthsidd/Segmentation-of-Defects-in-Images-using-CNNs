import keras
import numpy as np
import os, sys, math
from config.constants import *

def get_batch(splitName='train',
                    chunkPath=train_name_fmt, 
                    isShuffle=False,
                    chunkCount=0, 
                    n_channels=3):
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

    while(1):
      
      if isShuffle is True:
          np.random.shuffle(indexes)

      for i in indexes:
          chunk_data_path = os.path.join(data_path,
                                         chunkPath.format(splitName, i, 'x'))
          chunk_label_path = os.path.join(data_path,
                                          chunkPath.format(splitName, i, 'y'))
          with open('./VGGProcessing.log','w') as fh:
            fh.write('Processing : {}'.format(chunk_data_path))
          
          data = np.load(chunk_data_path)
          data_1 = np.copy(data)
          label = np.load(chunk_label_path)

          label = keras.utils.to_categorical(label, num_classes)
          
          #Relicate to three channels
          if n_channels == 3:
            data = np.repeat(data, n_channels, axis=1)
          
          #Transpose the data
          data = np.transpose(data, (0, 2 ,3, 1))

          num_of_minibatches = math.ceil(data.shape[0] / batch_size) - 1

          for k in range(num_of_minibatches):
            yield (data[k*batch_size:(k+1) * batch_size],label[k*batch_size: (k+1) * batch_size])

          #Yield the remaining samples
          if len(data[num_of_minibatches*batch_size:]) > 0:
            yield (data[num_of_minibatches*batch_size:],label[num_of_minibatches*batch_size:])


def gen_for_GAP_data(split_name, n_channels):
  
  if split_name == "valid":
    gen = get_batch(split_name, 
                    valid_name_fmt,
                    isShuffle=False,
                    chunkCount=valid_chunk_count,
                    n_channels=n_channels)
    steps = valid_batches

  elif split_name == "test":
    gen = get_batch(split_name,
                    test_name_fmt,
                    isShuffle=False,
                    chunkCount=test_chunk_count,
                    n_channels=n_channels)    
    steps = test_batches
    
  elif split_name == "train":
    gen = get_batch(split_name,
                    train_name_fmt,
                    isShuffle=False,
                    chunkCount=train_chunk_count,
                    n_channels=n_channels)   
    steps = train_batches
    
  return gen, steps
  