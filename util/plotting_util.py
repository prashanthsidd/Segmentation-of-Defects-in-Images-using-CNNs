import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

def get_roc_curve(labels, predictions, num_classes):
  fpr = dict()
  tpr = dict()
  thresholds = dict()
  roc_auc = dict()

  for i in range(num_classes):
      fpr[i], tpr[i], thresholds[i] = roc_curve(labels, predictions)
      roc_auc[i] = auc(fpr[i], tpr[i])    


  fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(labels.ravel(), predictions.ravel())
  roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

  plt.figure()
  lw = 2
  plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[1])
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver operating characteristic example')
  plt.legend(loc="lower right")
  plt.show()
  
  return fpr, tpr, thresholds, roc_auc


def get_pr_curve(labels,  predictions):
  precision, recall, _ = precision_recall_curve(labels, predictions)

  plt.step(recall, precision, color='b', alpha=0.2,
    where='post')
  plt.fill_between(recall, precision, step='post', alpha=0.2,
                   color='o')

  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])

  return precision, recall