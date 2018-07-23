import numpy as np
import keras.backend as K
import tensorflow as tf


def c_mat(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * K.cast(tf.boolean_mask(label_true,mask), 'int') +
        tf.boolean_mask(label_pred,mask), minlength=n_class ** 2)
    
    hist = K.reshape(hist, (n_class, n_class))
    return hist

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


def precision(y_true, y_pred):
    """Compute precision"""
    confusion = compute_error_matrix(y_true, y_pred)
    per_class_precision = K.cast(tf.diag_part(confusion), 'float') / K.cast(K.sum(confusion, axis=-1), 'float')
    mask = tf.logical_not(tf.is_nan(per_class_precision))
    per_class_precision = tf.boolean_mask(per_class_precision, mask)
    return K.mean(per_class_precision)

def recall(y_true, y_pred):
    """Compute recall"""
    confusion = compute_error_matrix(y_true, y_pred)
    per_class_recall = K.cast(tf.diag_part(confusion), 'float') / K.cast(K.sum(confusion, axis=0), 'float')
    mask = tf.logical_not(tf.is_nan(per_class_recall))
    per_class_recall = tf.boolean_mask(per_class_recall, mask)
    return K.mean(per_class_recall)
  

def f1(y_true, y_pred):
  """Compute accuracy"""
  pres = precision(y_true, y_pred)
  recl = recall(y_true, y_pred)
  
  return 2 *  ((pres * recl) / (pres + recl))