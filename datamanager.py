import torch
import torchvision.transforms as transforms

from os.path import isfile
from random import shuffle







class EMNIST_Dataset(torch.utils.data.Dataset):
	def __init__(self, data, labels, apply_transform=False):
		assert data.shape[0] == labels.shape[0]
		
		self.data = data
		self.labels = labels

		if apply_transform:
			self.transform = transforms.Compose([
									transforms.ToPILImage(),
									transforms.RandomRotation(10),
									transforms.Pad(2),
									transforms.RandomCrop(28),
									transforms.ToTensor()
								])
		else:
			self.transform = transforms.Compose([
									transforms.ToPILImage(),
									transforms.ToTensor()
								])




	def __getitem__(self, idx):
		

		return ( self.transform(self.data[idx]), self.labels[idx] )
		

	def __len__(self):
		return self.labels.shape[0]





def create_ensemble_training_sets(directory="", validation_size=0.1, k=5):
	data = torch.load(directory+"train-images.pt")
	labels = torch.load(directory+"train-labels.pt")
	l = labels.shape[0] # Length of dataset

	# Generate a list of indices shuffled around 
	# for ease of random split
	indices = [i for i in range(l)] 
	shuffle(indices)
	indices = torch.tensor(indices)

	# Step 
	step = int( l * validation_size )
	indices_blocks = []

	for i in range(k):
		train_set = torch.cat( 
						(
							indices[0:i*step],
							indices[(i+1)*step:] 
						), 
						dim=0
					)

		validation_set = indices[i*step:(i+1)*step] 

		# Append a tuple containing the indices of
		# a training set and a validation set
		indices_blocks.append( (train_set, validation_set) )


	datasets = []

	for i in range(k):
		datasets.append(
			(
				EMNIST_Dataset(data[ indices_blocks[i][0] ], labels[ indices_blocks[i][0] ], apply_transform=True), # Training set
				EMNIST_Dataset(data[ indices_blocks[i][1] ], labels[ indices_blocks[i][1] ]), # Validation set
			)
		)

	return datasets

	

def get_ensemble_training_sets(directory="", validation_size=0.1, k=5, save_load=True):
	# Try to load already created training sets
	if save_load and isfile("ensemble_training_sets.pt"):
		return torch.load("ensemble_training_sets.pt")

	datasets = create_ensemble_training_sets(directory, validation_size, k)

	# Save the training set for later use
	if save_load:
		torch.save(datasets, "ensemble_training_sets.pt")

	return datasets



def get_test_data(directory=""):
	data = torch.load(directory+"test-images.pt")
	labels = torch.load(directory+"test-labels.pt")

	return EMNIST_Dataset(data=data, labels=labels)


def get_datasets(directory=""):
	train_data = torch.load(directory+"train-images.pt")
	train_labels = torch.load(directory+"train-labels.pt")

	test_data = torch.load(directory+"test-images.pt")
	test_labels = torch.load(directory+"test-labels.pt")

	return EMNIST_Dataset(data=train_data, labels=train_labels, apply_transform=True), EMNIST_Dataset(data=test_data, labels=test_labels)





