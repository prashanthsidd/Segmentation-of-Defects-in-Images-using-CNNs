from keras import load_model
from metrics.Metrics import mean_IU
from loss import weighted_pixelwise_crossentropy
from util.BilinearUpSampling import BilinearUpSampling2D

def load_model_with_weights(model_weights_path, class_weights=None):

    custom_objects = {"pixelwise_crossentropy": weighted_pixelwise_crossentropy(class_weights),
                      "BilinearUpSampling2D": BilinearUpSampling2D,
                      'mean_IU': mean_IU}

    return load_model(model_weights_path, custom_objects=custom_objects)
