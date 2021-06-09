import torch 
import torch.nn.functional as F

from time import time
import numpy as np


# Takes a model and a loader and tests the accuracy 
# of the model on the dataset wrapped in the loader
def test_model(net, loader):
	total = 0.0
	correct = 0.0


	for i, (data, labels) in enumerate(loader):
		# Load data into GPU using cuda
		data = data.cuda()

		# Get the model's predictions
		outputs = F.softmax( net(data).detach(), dim=1 )

		# Find which class the model thinks is most likely
		predicted = torch.argmax(
						outputs,
						dim=1
					).cpu().float() 
			
		# Compare the model's predictions with the actual
		# labels and sum up the number of correct predictions
		correct += torch.sum( predicted == labels ).item()

		# Update the total number of datapoints processed
		total += labels.shape[0]


	return correct / total


# Takes an ensemble and a loader and tests the accuracy 
# of the ensemble on the dataset wrapped in the loader
def test_ensemble(ensemble, loader):
	total = 0.0
	correct = 0.0


	for i, (data, labels) in enumerate(loader):
		# Load data into GPU using cuda
		data = data.cuda()

		# Get the ensemble's predictions
		outputs = F.softmax( ensemble.forward(data), dim=1 )

		# Find which class the ensemble thinks is most likely
		predicted = torch.argmax(
						outputs,
						dim=1
					).cpu().float() 
			
		# Compare the ensemble's predictions with the actual
		# labels and sum up the number of correct predictions
		correct += torch.sum( predicted == labels ).item()

		# Update the total number of datapoints processed
		total += labels.shape[0]


	return correct / total









