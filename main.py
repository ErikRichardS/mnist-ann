import torch 
import torchvision
#import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from os.path import isfile
from os import remove
from random import shuffle


from ann import VGG, Ensemble
from datamanager import *
from trainer import train_model
from tester import test_model, test_ensemble





def reset():
	if isfile("ensemble_training_sets.pt"):
		remove("ensemble_training_sets.pt")
	if isfile("checkpoint.pt"):
		remove("checkpoint.pt")






k = 5
directory = "data/digits/"


# Create k different training and validation datasets from the 
# training set. The validation sets are exclusive, thus all 
# datapoints will be used in the direct training of k-1 models. 
# 
# Order of nestled lists and tuples:
# 0 - List of dataset split
# 1 - Tuple of train and validation sets
# 2 - List of datapoints
# 3 - Tuple of data and label 
datasets = get_ensemble_training_sets(directory=directory, k=k)



# Create a list of trained models for the ensemble
model_list = []

for i in range(k):
	save_file = "model"+str(i)+".pt"
	if isfile(save_file):
		model_list.append( torch.load(save_file) )

	else:
		net = train_model( VGG(), datasets[i][0], datasets[i][1] )
		torch.save(net, save_file)

		model_list.append( net )






# Create the ensemble from the list of trained models
ensemble = Ensemble(model_list)

# Load test set and test the ensemble on it. 
tst_dataset = get_test_data(directory=directory)
tst_loader = DataLoader(tst_dataset, batch_size=100)
print( test_ensemble(ensemble, tst_loader) )
