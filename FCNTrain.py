import os
import math

import keras
from keras.models import load_model

from models.DD_FCN2 import DD_2s
from models.DD_FCN4 import DD_4s
from models.DD_FCN8 import DD_8s
from models.DD_FCN16 import DD_16s
from models.DD_FCN32 import DD_32s

from loss import weighted_pixelwise_crossentropy
from class_weights import get_salumn1_class_weights
from util.BilinearUpSampling import BilinearUpSampling2D
from data_generators.SegDataGenerators import Segmentation_Generator
from metrics.Metrics import (accuracy,
                             mean_accuracy,
                             freq_weighted_IU,
                             mean_IU, compute_error_matrix
                             )

def train_model(model_type,
                weights_path,  
                train_images_path, 
                train_GT_path, 
                val_images_path, 
                val_GT_path, 
                input_width,
                input_height,
                lr,
                freeze_deconv_ly,
                save_path,
                validate=False,
                initial_epoch=0,
                num_of_epochs=10,
                batch_size=1,
                n_classes=2,
                class_weights_idx='2',
                optimizer='adadelta', 
                resume_training=False,):
  
  dd_FCNs = { 'FCN2s':DD_2s, 'FCN16s':DD_16s, 'FCN8s':DD_8s, 'FCN32s':DD_32s, 'FCN4s':DD_4s}
  
  tot_train_smpls =  math.ceil(len(os.listdir(train_images_path)) / batch_size)
  tot_val_smpls = math.ceil(len(os.listdir(val_images_path)) / batch_size)
  
  assert model_type in dd_FCNs.keys(), \
    "Model name must be one of the following %r" % dd_FCNs.keys()

  if weights_path is not None and resume_training is False:
    
    m = dd_FCNs[model_type](None, input_shape=(input_height, 
                                               input_width,
                                               3), 
                            n_classes=n_classes)
    

    m.compile(optimizer= keras.optimizers.adam(lr=lr),
             loss=weighted_pixelwise_crossentropy(class_weights_idx),
             metrics=['acc', mean_IU])
    
    m.load_weights(weights_path, by_name=True, reshape=True)
      
    if freeze_deconv_ly is True:
      for layer in m.layers:
        layer.trainable = False

      m.layers[22].trainable = True
    
  else:
    
    custom_objects = {"pixelwise_crossentropy": weighted_pixelwise_crossentropy(class_weights_idx),
     "BilinearUpSampling2D": BilinearUpSampling2D, 'mean_IU': mean_IU}

    m = load_model(weights_path, custom_objects=custom_objects)
    
  m.summary()
     
 
  train_generator = Segmentation_Generator(train_images_path, 
                                           train_GT_path, 
                                           batch_size, 
                                           n_classes, 
                                           input_width, 
                                           input_height, 
                                           input_height, 
                                           input_width)
  
#   print(next(train_generator)[0].shape)
  
  if validate == True:
    
    val_generator = Segmentation_Generator(val_images_path, 
                                           val_GT_path, 
                                           batch_size, 
                                           n_classes, 
                                           input_width, 
                                           input_height, 
                                           input_height, 
                                           input_width)
   
  
  #callback list 
  callbacks = []
  tensorboard = TensorBoard(log_dir='{}/logs'.format(save_path), 
                            batch_size=1)
  
  callbacks.append(tensorboard)
  
  csv_logger = CSVLogger('{}/training.log'.format(save_path), append=True)
  
  callbacks.append(csv_logger)
  
  checkpoint = ModelCheckpoint(save_path+'{epoch:02f}.hdf5')
  
  callbacks.append(checkpoint)
  
  if not validate:

    m.fit_generator(train_generator,
                        tot_train_smpls,
                        initial_epoch=initial_epoch,
                        epochs=initial_epoch + num_of_epochs,
                        callbacks=callbacks)
    
  else:

    m.fit_generator(train_generator,
                    tot_train_smpls, 
                    validation_data=val_generator,
                    validation_steps=tot_val_smpls,
                    initial_epoch=initial_epoch,
                    callbacks=callbacks,
                    epochs=initial_epoch + num_of_epochs)