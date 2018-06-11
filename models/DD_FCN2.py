import keras
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPool2D, Input, Activation, Add
from keras.applications import VGG16
from util.BilinearUpSampling import BilinearUpSampling2D 
from util.Cropping import crop

def DD_2s(weights, input_shape=(64,64,3)):

  #Input shape
  if input_shape is not None:
    img_inp = Input(shape=input_shape, name='input_1')
  else:
    img_inp = Input(shape=(None, None, 3), name='input_1')
  
  #Block 1
  x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv1', strides=(1,1))(img_inp)
  x = Conv2D(64, (3,3), activation='relu', padding='same', name='block1_conv2')(x)
  x= MaxPool2D(strides=(2,2), name='block1_pool')(x)
  p1 = x
  
  #Block 2
  x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv1')(x)
  x = Conv2D(128, (3,3), activation='relu', padding='same', name='block2_conv2')(x)
  x = MaxPool2D(strides=(2,2), name='block2_pool')(x)
  p2 = x 
  
  #Block 3
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv1')(x)
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv2')(x)
  x = Conv2D(256, (3,3), activation='relu', padding='same', name='block3_conv3')(x)
  x = MaxPool2D(strides=(2,2), name='block3_pool')(x)
  p3 = x
  
  #Block 4
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv1')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv2')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block4_conv3')(x)
  x = MaxPool2D(strides=(2,2), name='block4_pool')(x)
  p4 = x
  #Block 5
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv1')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv2')(x)
  x = Conv2D(512, (3,3), activation='relu', padding='same', name='block5_conv3')(x)
  x = MaxPool2D(strides=(2,2), name='block5_pool')(x)
  p5 = x

  #FC layers as convolutional layers
  x = Conv2D(1000, (2,2), activation='relu', padding='same', name='dense_1', kernel_regularizer=l2(0.001))(x)
  x = Conv2D(500, (1,1), activation='relu', padding='same', name='dense_2', kernel_regularizer=l2(0.001))(x)
  
  #32s
  x = Conv2D(2, (1,1), activation='linear', padding='valid', name='dense', kernel_initializer='he_normal')(x)
  
  o = BilinearUpSampling2D(size=(2,2))(x)
  
  #16s
  o2 = p4
  
  o2 = Conv2D(2, (1,1), kernel_initializer='he_normal')(o2)
  
  o,o2  = crop(o, o2, img_inp)
  
  o = Add()([o,o2])
  
  o = BilinearUpSampling2D(size=(2,2))(o)
  
  #8s 
  o2 = p3
  
  o2 = Conv2D(2, (1,1), kernel_initializer='he_normal')(o2)
  
  o2, o = crop(o2,o, img_inp)
	
  o = Add()([o,o2])
  
  o = BilinearUpSampling2D(size=(2,2))(o)
  
  #4s
  o2 = p2
  
  o2 = Conv2D(2, (1,1), kernel_initializer='he_normal')(o2)
  
  o2, o = crop(o2,o, img_inp)
	
  o = Add()([o,o2])
  
  o = BilinearUpSampling2D(size=(2,2))(o)
  
  #2s
  o2 = p1
  
  o2 = Conv2D(2, (1,1), kernel_initializer='he_normal')(o2)
  
  o2, o = crop(o2,o, img_inp)
	
  o = Add()([o,o2])
  
  o = BilinearUpSampling2D(size=(2,2))(o)
  
  
  o_shape = Model(img_inp , o ).output_shape
  
  print("output shape is {}".format(o_shape))
  
  output_height = o_shape[2]
  output_width = o_shape[1]
   
  o = Activation(activation='softmax')(o)
  
  model = Model(img_inp, o)
  
  model.outputWidth = output_width
  model.outputHeight = output_height
  
  if(weights is not None):
    model.load_weights(weights, by_name=True,)
    
  return model