import torch 
import torch.nn as nn
from torch.utils.data import DataLoader


from time import time
from os.path import isfile
from os import remove
from copy import deepcopy



from settings import hyperparameters
from tester import test_model


checkpoint_path = "checkpoint.pt"


def checkpoint_exists():
	return isfile( checkpoint_path )


def save_checkpoint(net, epochs_left, learning_rate, best_model, best_accuracy):
	torch.save({
				"model-state-dict" : net.state_dict(),
				"epoch" : epochs_left,
				"learning-rate" : learning_rate,
				"best-model-dict" : best_model.state_dict(),
				"accuracy" : best_accuracy,
			}, checkpoint_path )


def load_checkpoint(net, best_model):
	checkpoint = torch.load( checkpoint_path )
	net.load_state_dict(checkpoint["model-state-dict"])
	num_epochs = checkpoint["epoch"]
	learning_rate = checkpoint["learning-rate"]
	best_model.load_state_dict(checkpoint["best-model-dict"])
	best_accuracy = checkpoint["accuracy"]
	
	return net, num_epochs, learning_rate, best_model, best_accuracy


def delete_checkpoint():
	if isfile( checkpoint_path ):
		remove( checkpoint_path ) 



# Trains a NN model. 
# Returns 1 if it starting to overfit before all epochs are done
# Returns 0 otherwise
def train_model(net, trn_dataset, vld_dataset, start_fresh=False):

	# Hyper Parameters
	num_epochs = hyperparameters["number-epochs"]
	batch_size = hyperparameters["batch-size"]
	learning_rate = hyperparameters["learning-rate"]
	wt_decay = hyperparameters["weight-decay"]
	lr_decay = hyperparameters["learning-decay"]


	# Wrap the datasets in loaders that will handle the fetching of data and labels
	# batch_size - How many datapoints is loaded each fetch.
	# shuffle - Randomize the order loaded datapoints. 
	trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
	vld_loader = torch.utils.data.DataLoader(vld_dataset, batch_size=batch_size)


	# Loss calculates the error of the output
	# Optimizer does the backprop to adjust the weights of the NN
	criterion = nn.CrossEntropyLoss() 
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


	

	best_accuracy = 0
	best_model = deepcopy(net)

	# Check if checkpoint of saved progress exists. 
	# If it does, load and continue from saved checkpoint. 
	if not start_fresh and checkpoint_exists():
		print("Loading progress from last successful epoch...", end="\r")
		net, num_epochs, learning_rate, best_model, best_accuracy = load_checkpoint(net, best_model)
		optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
		print("Progress loaded. Continuing training from last successful epoch.")
		print("Accuracy :\t %0.3f" % (best_accuracy))
		print("Learning rate :\t %f" % (learning_rate))
		print("Epochs left :\t %d" % (num_epochs))

	
	# Train the Model
	print("Begin training...")
	for epoch in range(num_epochs):
		t1 = time()

		loss_sum = 0

		for i, (data, labels) in enumerate(trn_loader):
			# Load data into GPU using cuda
			data = data.cuda()
			labels = labels.cuda().long()

			# Forward + Backward + Optimize
			optimizer.zero_grad()
			outputs = net(data)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			loss_sum += loss
			
				
		t2 = time()
			

		print("Epoch time : %0.3f m \t Loss : %0.3f" % ( (t2-t1)/60 , loss_sum ))

		# Test new iteration of the model on the validation set.
		print("Testing data..", end="\r")
		validation_accuracy = test_model(net, vld_loader)
		print("Testing is done.")
		print("Current validation accuracy : %f" % (validation_accuracy))


		# Reduce learning rate for next epoch
		learning_rate *= (1-lr_decay)
		optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate) #, weight_decay=wt_decay)
		print("New learning rate : %f" % (learning_rate))

		# If accuracy has improved: Set current model as best model.
		if validation_accuracy > best_accuracy:
			best_model = deepcopy(net)
			best_accuracy = accuracy


		# Save training progress
		print("Saving training progress...", end="\r")
		save_checkpoint(
				net=net, 
				epochs_left=num_epochs-epoch-1, 
				learning_rate=learning_rate, 
				best_model=best_model, 
				best_accuracy=best_accuracy
			)
		print("Progress has been saved. Epoch %d of %d done." % (epoch+1, num_epochs))
		print()

		
	delete_checkpoint()
	return best_model













