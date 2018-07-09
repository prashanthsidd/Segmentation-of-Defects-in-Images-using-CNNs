import os
import cv2 as cv
import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

  
  
def seg_metrics_from_conf_matrix(conf_m):
  
    #accuracy
    acc = np.diag(conf_m).sum() / np.sum(conf_m)
    
    #mean accuracy 
    per_cls_acc = np.diag(conf_m) / np.sum(conf_m, axis=1)
    mean_acc = np.mean(per_cls_acc)
  
    
    
    #Precision
    per_cls_precision = np.diag(conf_m) / np.sum(conf_m, axis=0)
    mask = np.isfinite(per_cls_precision)
    precision = np.mean(per_cls_precision[mask])
    
    #Recall
    per_cls_recall = np.diag(conf_m) / np.sum(conf_m, axis=1)
    mask = np.isfinite(per_cls_recall)
    recall = np.mean(per_cls_recall[mask])
    
    #F1
    f1 = 2 * (precision * recall) / (precision + recall)
    
    #IOU and meanIOU
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    meanIOU = np.mean(IOU)
    
    #Frequency weighted IoU
    freq = np.sum(conf_m, axis=1) / conf_m.sum()
    fwIoU = np.sum(freq[freq > 0] * IOU[freq > 0])
    
    
    return acc, mean_acc, precision, recall, f1, meanIOU, fwIoU
  

def calculate_seg_metrics_from_predictions(nb_classes, res_dir, label_dir):
    conf_m = np.zeros((nb_classes, nb_classes), dtype=float)
    image_list = os.listdir(res_dir)
    for img_name in image_list:
        
        pred = cv.imread(os.path.join(res_dir, img_name), 0)
        pred[pred >= nb_classes] = 0

        label = cv.imread(os.path.join(label_dir, img_name), 0)

        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
    
        conf_m += _fast_hist(flat_label, flat_pred, nb_classes)


    return seg_metrics_from_conf_matrix(conf_m)