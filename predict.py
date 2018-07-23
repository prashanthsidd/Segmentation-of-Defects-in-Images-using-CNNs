import os
import cv2 as cv
import numpy as np
from data_generators.SegDataGenerators  import Preprocess_Images
from util.load_model_with_weights import load_mod_with_diff_dimensions

def get_predictions(model_type,
                    model_weights, 
                    data_path,
                    save_path,
                    n_classes=2,
                    height = 128,
                    width = 128,
                    thresh=0.6):
  
  #Load model 
  
  m = load_mod_with_diff_dimensions(model_type, 
                                    model_weights, 
                                    height, 
                                    width, 
                                    n_classes)
  print(m.metrics_names)
  
  files = os.listdir(data_path)
    
  for file in files:

    im = Preprocess_Images(os.path.join(data_path, file), 
                           width, 
                           height)

    im = np.expand_dims(im, axis=0)

    result = m.predict(np.array(im), batch_size=1)

    result[result < thresh] = 0

    result = np.argmax(np.squeeze(result), axis=-1).astype(np.uint8)

    if not os.path.exists(save_path):
      os.makedirs(save_path)

    cv.imwrite(os.path.join(save_path + file[:-4] + '.png'), result)