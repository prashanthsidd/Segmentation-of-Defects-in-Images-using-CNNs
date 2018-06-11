from keras import Model
from keras.layers import Cropping2D

def crop( o1 , o2 , i  ):
  o_shape2 = Model( i  , o2 ).output_shape
  outputHeight2 = o_shape2[2]
  outputWidth2 = o_shape2[3]

  o_shape1 = Model( i  , o1 ).output_shape
  outputHeight1 = o_shape1[2]
  outputWidth1 = o_shape1[3]

  cx = abs( outputWidth1 - outputWidth2 )
  cy = abs( outputHeight2 - outputHeight1 )

  if outputWidth1 > outputWidth2:
    o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )))(o1)
  else:
    o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )))(o2)

  if outputHeight1 > outputHeight2 :
    o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )))(o1)
  else:
    o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )))(o2)
  return o1 , o2 