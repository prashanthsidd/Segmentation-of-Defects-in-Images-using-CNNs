import os, sys
import numpy as np
from time import time

import keras
from keras.optimizers import SGD
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

from models import asinvos
from config.constants import *
from data_generators.batch_generators import get_batch

"""
Train the models
"""

if __name__ == '__main__':
    #Callbacks to be used
    checkpoint = ModelCheckpoint(checkpoint_fmt, 
                                 verbose=1,
                                 monitor='loss')

    tensorboard = TensorBoard(log_dir='{}/logs/{}'.format(save_dir,
                                                          time()),
                              batch_size=batch_size)

    csvLogger = CSVLogger("{}/training.log".format(save_dir),
                          append=True)
    
    callback_list = [checkpoint, tensorboard, csvLogger]

    #Start training
    model = asinvos.get_model()
    model.summary()

    history = model.fit_generator(
        generator=get_batch('train', 
                                train_name_fmt,
                                chunkCount=train_chunk_count),
        steps_per_epoch=train_batches,
        callbacks=callback_list,
        validation_data=get_batch('valid', 
                                      valid_name_fmt,
                                      chunkCount=valid_chunk_count),
        validation_steps=valid_batches,
        epochs=10)

    
    #Save history?
    np.save(history.history, "{}/latest_model.hdf5".format(save_dir))

    #TODO Metrics check??

    print("")
