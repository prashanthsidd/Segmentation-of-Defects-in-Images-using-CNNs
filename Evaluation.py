import os
import numpy as np
from glob import glob

from util.load_model_with_weights import load_model_with_weights
from data_generators.SegDataGenerators import Segmentation_Generator
from metrics.globa_metrics import seg_metrics_from_conf_matrix, _fast_hist

def evaluate_models(models_weights_path, 
                    n_class, 
                    imgs_path,
                    gt_path,
                    thresh,
                    batch_size,
                    height,
                    width,
                    class_weights,
                    fn_prefix):

    '''Utility to calculate global metrics for all the models under models_weights_path.'''

    models_weights = glob(models_weights_path + '*.hdf5')

    seg_metrics_list = []

    for weight in models_weights:

        print('Processing ::', weight)
        try:
            m = load_model_with_weights(weight, class_weights)

            print('\t Generating predictions ::')

            hist = np.zeros((n_class, n_class))

            steps = len(os.listdir(imgs_path)) // batch_size

            eval_generator = Segmentation_Generator(imgs_path,
                                                    gt_path,
                                                    batch_size,
                                                    n_class,
                                                    width,
                                                    height,
                                                    height,
                                                    width)
            for i in range(steps):
                X, Y_true = next(eval_generator)

                Y_pred = m.predict_on_batch(X)
                Y_pred[Y_pred < thresh] = 0
                Y_true = np.argmax(Y_true, axis=-1).astype(np.uint8)
                Y_true[Y_true >= n_class] = 0

                Y_pred = np.argmax(Y_pred, axis=-1).astype(np.uint8)
                Y_pred[Y_pred >= n_class] = 0

                for lt, lp in zip(Y_true, Y_pred):
                    hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)


            print('\t Calculating metrics ::')

            metrics = list(seg_metrics_from_conf_matrix(hist))
            metrics.insert(0, np.int(weight.split('/')[-1].split('.')[0]))

            seg_metrics_list.append(metrics)

            print('\t\t', seg_metrics_list, '\n\n')

            del m

        except Exception as e:
            print(e)

    seg_metrics_list = np.array(seg_metrics_list)
    seg_metrics_list = seg_metrics_list[seg_metrics_list[:, 0].argsort()]

    np.savetxt(models_weights_path + fn_prefix + str(n_class) + '.log',
               seg_metrics_list, delimiter=',',
               header='epoch,accuracy,mean_accuracy,precision,recall,f1,mean_IU,freq_weighted_IU')

    return seg_metrics_list
