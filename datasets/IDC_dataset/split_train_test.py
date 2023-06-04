# Sample of the script to move the testing data from the complete data folder
# for the IDC dataset

import pickle

test_class_0 = []
test_class_1 = []

with open('./bh_test_class_0.pkl', 'rb') as fp:
    test_class_0 = pickle.load(fp)

with open('./bh_test_class_1.pkl', 'rb') as fp:
    test_class_1 = pickle.load(fp)
    
    
# Make dir if needed


import shutil 

train_dir_0 = r"../data/0" #Full data directory
test_dir_0 = r"../test/0" # Testing dir for class 0
for file_name in test_class_0:
  source = os.path.join(train_dir_0,file_name)
  destination = os.path.join(test_dir_0,file_name)
  dest = shutil.move(source, destination) 

train_dir_1 = r"../data/1"
test_dir_1 = r"../test/1"
for file_name in test_class_1:
  source = os.path.join(train_dir_1,file_name)
  destination = os.path.join(test_dir_1,file_name)
  dest = shutil.move(source, destination) 