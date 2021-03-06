import keras
import cv2 as cv
import numpy as np
import math, os, random, itertools


# Read the images from the path

def Preprocess_Images(path, width, height):
  
  try:
    
    im = cv.imread(path)
   
    im = np.float32(im)
    im = np.float32(cv.resize(im, (width, height), interpolation=cv.INTER_NEAREST))
    
    im -= 127.5
    im /= 127.5
    
    return im
  
  except  Exception as e:
    print ("Preprocess img", path, e)
    im = np.zeros((height, width, 3))
    return im
  
  
def Preprocess_Segmentation(path, width, height, n_classes):
    
  try:
    
    seg = cv.imread(path, 0)
    
    seg = cv.resize(seg, (width,height), interpolation=cv.INTER_NEAREST)
    seg[seg >= n_classes] = 0
    seg = keras.utils.to_categorical(seg, n_classes)
    
    return seg
  
  except  Exception as e:
    print("Preprocess Seg", path, e)  


def Segmentation_Generator(images_path, 
                           seg_path, 
                           batch_size, 
                           n_classes, 
                           input_width, 
                           input_height, 
                           output_height, 
                           output_width,
                           shuffle=False):
  
  assert os.path.isdir(images_path) and os.path.isdir(seg_path)
  
  images = os.listdir(images_path)
  images.sort()
  
  segmentations = os.listdir(seg_path)
  segmentations.sort()
  
  assert len(images) == len(segmentations)
   
  img_seg_zip = zip(images,segmentations)

  zip_list = list(img_seg_zip)
  
  if shuffle == True:
    random.shuffle(zip_list)

  zipped = itertools.cycle(zip_list)
  
  count = 0
  
  while True:
    
    X = []
    Y = []

    if count >= math.ceil(len(images) / batch_size) and shuffle == True:
      
      zip_list = list(zip(images,segmentations))
      random.shuffle(zip_list)
      zipped = itertools.cycle(zip_list)
      count=0
    
    for _ in range(batch_size):
      im, seg = next(zipped)
      
      X.append(Preprocess_Images(os.path.join(images_path, im), 
                                 input_width, 
                                 input_height))
      
      Y.append(Preprocess_Segmentation(os.path.join(seg_path, seg), 
                                       output_width,
                                       output_height, n_classes))
    count += 1

    yield np.array(X),np.array(Y)   