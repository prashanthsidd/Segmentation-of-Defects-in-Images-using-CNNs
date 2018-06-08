import os
import cv2 as cv
import numpy as np

def data_aug(image_path, gt_path):
  
  imgs = os.listdir(image_path)
  counter = len(imgs)
  print('counter is {}'.format(counter))
  for img in imgs:
  
    gt_name = img[:-4] + '.png'
    
    im = cv.imread(os.path.join(image_path, img))  
    gt = cv.imread(os.path.join(gt_path, gt_name))  
    
    assert im is not None and gt is not None 
    
    #Flip
    im_flip = np.fliplr(im)
    gt_flip = np.fliplr(gt)
    
    cv.imwrite(os.path.join(image_path, 
                            img[:-4] + '_' + str(counter) + img[-4:]),
               im_flip)
      
    cv.imwrite(os.path.join(gt_path, 
                            img[:-4] + '_' + str(counter) + '.png'), 
               gt_flip)
    
    counter += 1
    
    #Rotate 90 degrees 3 times
    for i in range(3):
      
      im = np.rot90(im)
      gt = np.rot90(gt)
      
      cv.imwrite(os.path.join(image_path, 
                              img[:-4] + '_' + str(counter) + img[-4:]), 
                 im)
      
      cv.imwrite(os.path.join(gt_path, 
                              img[:-4] + '_' + str(counter) + '.png'), 
                 gt)
      
      counter += 1
      
      im_flip = np.rot90(im_flip)
      gt_flip = np.rot90(gt_flip)
      
      cv.imwrite(os.path.join(image_path, 
                              img[:-4] + '_' + str(counter) + img[-4:]), 
                 im_flip)
      
      cv.imwrite(os.path.join(gt_path, 
                              img[:-4] + '_' + str(counter) + '.png'), 
                 gt_flip)
      
      counter += 1