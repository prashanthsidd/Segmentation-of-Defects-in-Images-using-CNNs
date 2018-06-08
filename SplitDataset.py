import os
import random

def create_val_data(img_path, 
                    gt_path, 
                    test_img_path, 
                    test_gt_path,
                    val_img_path, 
                    val_gt_path, 
                    test_pcnt=10,
                    val_pcnt=10):
  
  '''Splits the given dataset, randomly, into train, test and validation sets.
     Paths will be created if they dont exist already. 
  
     Parameters:
       img_path : path to the dataset
       gt_path : path to ground truths of the dataset
       test_img_path : path to store test images
       test_gt_path : path to store  ground truths for the test images
       val_img_path : path to store validation images
       val_gt_path : path to store  ground truths for the validation images
       test_pcnt : percentage of the dataset to be randomly selected as testset
       test_pcnt : percentage of the dataset to be randomly selected as valdiationset
     
     Returns:
       None
  '''
  
  for i in [val_img_path, val_gt_path, test_img_path, test_gt_path]:
    
    if not os.path.exists(i):
      os.makedirs(i)
  
  files = os.listdir(img_path)

  tot_imgs = len(files)
  tot_val_imgs = tot_imgs * val_pcnt // 100
  tot_test_imgs = tot_imgs * test_pcnt // 100
  
  print(tot_test_imgs, tot_val_imgs)
  rand_list = random.sample(range(0, tot_imgs), 
                                tot_val_imgs + tot_test_imgs) 
  
  rand_test_list = rand_list[:tot_test_imgs]
  rand_val_list = rand_list[tot_test_imgs:]
  
  
  for i in rand_val_list:
    
    file = files[i]
    
    os.rename(os.path.join(img_path, file), 
              os.path.join(val_img_path, file))
    
    os.rename(os.path.join(gt_path, file[:-4] + ".png"), 
              os.path.join(val_gt_path, file[:-4] + ".png"))
    
    
  for i in rand_test_list:
    
    file = files[i]
    
    os.rename(os.path.join(img_path, file), 
              os.path.join(test_img_path, file))
    
    os.rename(os.path.join(gt_path, file[:-4] + ".png"), 
              os.path.join(test_gt_path, file[:-4] + ".png"))