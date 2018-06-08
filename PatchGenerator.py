import os
import math
import cv2 as cv
import numpy as np

def generate_patches(image_dir, 
                mask_dir, 
                save_dir_img, 
                save_dir_mask, 
                patch_dim=256):

  if not os.path.exists(save_dir_img):
    os.makedirs(save_dir_img,)

  if not os.path.exists(save_dir_mask):
    os.makedirs(save_dir_mask)

  #patch generation

  images = os.listdir(image_dir)
  counter = 0
  offset = patch_dim // 2
  
  for img_name in images:

    print('Processing : {}'.format(img_name) )
    
    img = cv.imread(os.path.join(image_dir, img_name))
    mask = cv.imread(os.path.join(mask_dir, img_name[:-4] + '.png'))

    h,w,_ = img.shape

    #resize to next nearest patch_dim divisible

    h = patch_dim * math.ceil(h / patch_dim)
    w = patch_dim * math.ceil(w / patch_dim)

    img = cv.resize(img, (w, h))
    mask = cv.resize(mask, (w, h))

    x_co = 0
    y_co = 0

    #Horizontal sweep
    while x_co + patch_dim <= w:

      #vertival sweep
      while y_co + patch_dim <= h:

        img_patch = img[y_co: y_co + patch_dim, x_co: x_co + patch_dim]
        mask_patch = mask[y_co: y_co + patch_dim, x_co: x_co + patch_dim]

        y_co = y_co + patch_dim // 2

        cv.imwrite(os.path.join(save_dir_img, 
                                img_name[:-4] + '_' + str(counter) + '.jpg'),
                   img_patch)

        cv.imwrite(os.path.join(save_dir_mask,
                                img_name[:-4] + '_' + str(counter) + '.png'),
                   mask_patch)

        counter += 1 

      y_co = 0
      #increase the x_co by 50% of the patch_dim
      x_co = x_co + offset

  counter = 0