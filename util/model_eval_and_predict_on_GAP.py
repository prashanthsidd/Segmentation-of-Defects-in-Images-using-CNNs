import sys
from data_generators.batch_generators import gen_for_GAP_data

def evaluate_model_on_GAP(model, split_name, n_channels):
  gen = None
  
  gen, steps = gen_for_GAP_data(split_name, n_channels)
  
  if gen is None: 
    print("split_name didnt match test or valid. Exiting!!!")
    sys.exit() 
    
  result = model.evaluate_generator(gen, steps=steps)

  print("Model metrics are :: " + str(model.metrics_names))
  print(result)
  
  return result