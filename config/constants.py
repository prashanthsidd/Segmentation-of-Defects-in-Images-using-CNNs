import os

#Dataset constants
test_chunk_count = 26
valid_chunk_count = 7
train_chunk_count = 154

num_test_instances = 810704
num_valid_instances = 201483
num_train_instances = 4897699

test_batches = 3167
valid_batches = 788
train_batches = 19132

data_path = "./data/"
valid_name_fmt = "{0}/chunks_64x64_NORMvsDISTRESS_{0}_chunk_{1}_{2}.npy"
test_name_fmt = "{0}/chunks_64x64_NORMvsDISTRESS_{0}_chunk_{1:02d}_{2}.npy"
train_name_fmt = "{0}/chunks_64x64_NORMvsDISTRESS_{0}_chunk_{1:03d}_{2}.npy"

#Training settings
batch_size = 256
num_classes = 2
epochs = 2
save_dir = os.path.join(os.getcwd(), 'saved_models/FC_2')
checkpoint_fmt = os.path.join(save_dir, 'best_model.hdf5')