import keras
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPool2D, Input, Activation
from keras.applications import VGG16
from util.BilinearUpSampling import BilinearUpSampling2D 

def DD_32s(weights, input_shape=(64,64,3)):

  #Input shape
  if input_shape is not None:
    img_inp = Input(shape=input_shape, name='input_1')
  else:
    img_inp = Input(shape=(None, None, 3), name='input_1')
  
  #Block 1
  x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1', strides=(1,1))(img_inp)
  x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
  x= MaxPool2D(strides=(2,2), name='block1_pool')(x)
  
  #Block 2
  x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPool2D(strides=(2,2), name='block2_pool')(x)
  
  #Block 3
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPool2D(strides=(2,2), name='block3_pool')(x)
  
  #Block 4
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPool2D(strides=(2,2), name='block4_pool')(x)

  #Block 5
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPool2D(strides=(2,2), name='block5_pool')(x)

  #FC layers as convolutional layers
  x = Conv2D(1000, (2,2), activation='relu', padding='same', name='dense_1', kernel_regularizer=l2(0.001))(x)
  x = Conv2D(500, (1,1), activation='relu', padding='same', name='dense_2', kernel_regularizer=l2(0.001))(x)
  x = Conv2D(2, (1,1), activation='linear', padding='valid', name='dense', kernel_initializer='he_normal')(x)
  
  o = BilinearUpSampling2D(size=(32,32))(x)
  
#   o = Conv2DTranspose(2, kernel_size=(64, 64), strides=(32,32), use_bias='False')(x)
  
#   o = UpSampling2D(size=(32,32))(x)
  
  o_shape = Model(img_inp , o ).output_shape
	
  print("output shape is {}".format(o_shape))
  
  output_height = o_shape[2]
  output_width = o_shape[1]
  
# #   bef_reshape = o
# #   print("Before reshape{}".format(o_shape))

#   o = Reshape((output_height*output_width, -1))(o)
  
#   print("After reshape{}".format(Model(img_inp , o ).output_shape))
  
  o = Activation(activation='softmax')(o)
  
  model = Model(img_inp, o)
  
  model.outputWidth = output_width
  model.outputHeight = output_height 
   
  return model