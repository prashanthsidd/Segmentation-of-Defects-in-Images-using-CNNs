import os
import cv2 as cv
import numpy as np

defect_classes = [1, 3, 5, 6, 7, 8, 12, 15, 17, 20, 22, 24, 26, 27, 28, 29, 30, 31, 38, 39, 41]

def generateTwoClassSegMaps(path_to_imgs, 
  path_to_def_files, 
  save_path, 
  header_lines_skip=7, 
  ignore_defects=[26, 27, 30]):
  
  if not os.path.exists(save_path):
    os.makedirs(save_path)
      
  img_sizes = []

  images = os.listdir(path_to_imgs) 

  # for each image read the corresponding defect file 
  # and generate segmentation map

  for img in images:
    
    im = cv.imread(os.path.join(path_to_imgs, img),1)
    print(img)
    img_sizes.append(im.shape)
    
    defects = np.genfromtxt(
        os.path.join(path_to_def_files, img[:-4] + ".def"),
        skip_header=7, 
        delimiter=' ', 
        usecols=(0,1,2,3,4), 
        dtype='int')
    
    mask = np.zeros(im.shape, dtype='uint8')
    
    for defect in defects:

      x1,y1,x2,y2,def_type = defect

      #mask for two class segmentation. 
      if def_type not in ignore_defects:
        mask[y1:y2, x1:x2] = 1
            
    cv.imwrite(os.path.join(save_path, img[:-4] + '.png'), mask)



def generateMultiClassSegMaps(path_to_imgs, 
  path_to_def_files, 
  save_path,
  num_of_classes=21, #including background 
  header_lines_skip=7, 
  ignore_defects=[27]):
  
  if not os.path.exists(save_path):
    os.makedirs(save_path)
      
  img_sizes = []

  images = os.listdir(path_to_imgs) 

  # for each image read the corresponding defect file 
  # and generate segmentation map

  for img in images:
    
    im = cv.imread(os.path.join(path_to_imgs, img),1)
    
    img_sizes.append(im.shape)
    
    defects = np.genfromtxt(
        os.path.join(path_to_def_files, img[:-4] + ".def"),
        skip_header=7, delimiter=' ', usecols=(0,1,2,3,4), dtype='int')
    
    #two dimentional mask is enough bcos data generators convert them to one hot vec
    mask = np.zeros((im.shape[0], im.shape[1]), dtype='uint8')
    
    for defect in defects:

      x1,y1,x2,y2,def_type = defect

      #mask for two class segmentation. 
      
      if def_type not in ignore_defects:
        idx = defect_classes.index(def_type)
        mask[y1:y2, x1:x2] = idx
      
    cv.imwrite(os.path.join(save_path, img[:-4] + '.png'), mask)