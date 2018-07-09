import tensorflow as tf

def weighted_pixelwise_crossentropy(cls_weights=[1]):

    def pixelwise_crossentropy(target, output):

        output = tf.clip_by_value(output, 10e-8, 1. - 10e-8)
        return - tf.reduce_sum(target * cls_weights * tf.log(output))

    return pixelwise_crossentropy
