import tensorflow as tf

def pixelwise_crossentropy(target, output):
  
  weights = [0.05, 0.95]
  output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
  return - tf.reduce_sum(target * weights * tf.log(output))