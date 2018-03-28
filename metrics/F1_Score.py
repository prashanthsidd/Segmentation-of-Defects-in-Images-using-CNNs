import os
from config.constants import *
from sklearn.metrics import f1_score
from data_generators.batch_generators import get_batch 

import keras
from keras.models import load_model, Sequential

def get_f1_score(model):
    
    """
    Function to calculate F1 score on test and validation data

    # Arguments
        model: Compiled keras model
    # Returns
        A dictionary of f1 scores on test and validation data
    """
    
    model = Sequential()
    predictions = model.predict_generator(
                    generator=get_batch('valid',
                                        valid_name_fmt,
                                        valid_chunk_count),
                    steps=valid_batches)

    #TODO check if any of the predictions is 0.5 
    yhat = (predictions > 0.5).astype('int32')
    f1_score(y, yhat) 