import keras
from keras.models import load_model
from class_weights import  get_salumn1_class_weights
from loss import weighted_pixelwise_crossentropy
from util.BilinearUpSampling import BilinearUpSampling2D
from models import DD_FCN2, DD_FCN4, DD_FCN8, DD_FCN16, DD_FCN32
from metrics.Metrics import mean_IU, accuracy, freq_weighted_IU
from metrics.Metrics import mean_accuracy, recall, f1, precision

dd_FCNs = {'FCN2s':DD_FCN2.DD_2s, 'FCN16s':DD_FCN16.DD_16s,
           'FCN8s':DD_FCN8.DD_8s, 'FCN32s':DD_FCN32.DD_32s,
           'FCN4s':DD_FCN4.DD_4s}

def load_model_with_custobjs(model_weights_path, class_weights=None):

    custom_objects = {"pixelwise_crossentropy": weighted_pixelwise_crossentropy(class_weights),
                      "BilinearUpSampling2D": BilinearUpSampling2D,
                      'mean_IU': mean_IU}

    return load_model(model_weights_path, custom_objects=custom_objects)


def load_mod_with_diff_dimensions(model_type,
                                  model_weights,
                                  height,
                                  width,
                                  n_classes):
    ''' Method to load weights into models with different spatial dimensions of input'''
    #Get class weights
    cls_weights = get_salumn1_class_weights(n_classes)
    
    m = dd_FCNs[model_type](None, (height,width,3),n_classes)
    
    m.compile(optimizer= keras.optimizers.SGD(),
                    loss=weighted_pixelwise_crossentropy(cls_weights),
                    metrics=[accuracy,
                            mean_accuracy,
                            mean_IU,
                            freq_weighted_IU,
                            precision,
                            recall,
                            f1])
    
    m.load_weights(model_weights)
    
    return m