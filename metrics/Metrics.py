import keras.backend as K
import tensorflow as tf

def compute_error_matrix(y_true, y_pred):
    """Compute Confusion matrix (a.k.a. error matrix).
    a       predicted
    c       0   1   2
    t  0 [[ 5,  3,  0],
    u  1  [ 2,  3,  1],
    a  2  [ 0,  2, 11]]
    l
    Note true positves are in diagonal
    """
    # Find channel axis given backend
        
    if K.image_data_format() == 'channels_last':
        ax_chn = 3
    else:
        ax_chn = 1
    
    classes = K.shape(y_true)[ax_chn]
    
    return tf.confusion_matrix(K.flatten(K.argmax(y_true, axis=ax_chn)), 
                               K.flatten(K.argmax(y_pred, axis=ax_chn)), 
                               num_classes = classes)

def accuracy(y_true, y_pred):
    """Compute accuracy."""
    confusion = compute_error_matrix(y_true, y_pred)
    # per-class accuracy
    acc = K.cast(K.sum(tf.diag_part(confusion)), 'float') / K.cast(K.sum(confusion), 'float')
    return acc
  
  
def mean_accuracy(y_true, y_pred):
    """Compute mean accuracy."""
    confusion = compute_error_matrix(y_true, y_pred)
    # per-class accuracy
    acc = tf.diag_part(confusion) / K.sum(confusion, axis=1)
    mask = tf.logical_not(tf.is_nan(acc))
    acc = tf.boolean_mask(acc, mask)
    return K.mean(acc)


def mean_IU(y_true, y_pred):
    """Compute mean IoU."""
    
    confusion = compute_error_matrix(y_true, y_pred)
    # per-class IU
    iu = K.cast(tf.diag_part(confusion) / (K.sum(confusion, axis=1) + K.sum(confusion, axis=0) - tf.diag_part(confusion)),
        'float')
    mask = tf.logical_not(tf.is_nan(iu))
    iu = tf.boolean_mask(iu, mask)
    return K.mean(iu)


def freq_weighted_IU(y_true, y_pred):
    """Compute frequent weighted IoU."""
    confusion = compute_error_matrix(y_true, y_pred)
    freq = K.cast(K.sum(confusion, axis=1), 'float') / K.cast(K.sum(confusion),'float')
    # per-class IU
    iu = K.cast(tf.diag_part(confusion) / (K.sum(confusion, axis=1) + 
                 K.sum(confusion, axis=0) - 
                 tf.diag_part(confusion)),
         'float')
    
    mask = tf.is_finite(freq) & tf.logical_not(tf.is_nan(iu))
    freq_masked = tf.boolean_mask(freq,mask)
    iu_masked = tf.boolean_mask(iu,mask)
    
    return K.sum(freq_masked * iu_masked)