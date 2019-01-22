import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from  torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *
import argparse
import dataset_reader
from torch.optim import lr_scheduler
import time
import os

def parse_args():

	# Argument parsing
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataroot', type=str, default='/home/meghana/Desktop/SKIN/skin-cancer-mnist-ham10000/', help='path to dataset containing csv files')
	parser.add_argument('--epochs', type=int, help='number of training epochs', default=10)
	parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
	return parser.parse_args()

def main():
	print("Start!")
	args = parse_args()
	
	#load dataset
	dataset = dataset_reader.import_train_dataset(args)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

	test_dataset = dataset_reader.import_test_dataset(args)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

	train_logger = SummaryWriter(log_dir = os.path.join('train'), comment = 'training')
	validation_logger = SummaryWriter(log_dir = os.path.join('validation'), comment = 'validation')

	#model
	n_class = 7
	num_epochs = args.epochs 
	num_epochs = int(num_epochs)

	# Load resnet network
	model = models.resnet50( pretrained=True )
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, n_class)

	# Freeze all layers
	for params in model.parameters():
		params.requires_grad = False

	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, n_class)

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")
	print("Using", device)
	model.to(device)  # Move model to device

	# Cross Entropy Loss
	# weights = [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
	# class_weights = torch.FloatTensor(weights).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.01, momentum=0.9)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	since = time.time()
	best_model_wts = model.state_dict()
	best_acc = 0.0
	count = 0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		scheduler.step()
		model.train(True)  # Set model to training mode
		running_loss = 0.0
		running_corrects = 0.0
		  # ttl num of correct predictions
		for inputs, metas in tqdm(dataloader):
			labels = Variable(metas).to(device)
			inputs = Variable(inputs).to(device)
			# Zero parameter gradients
			optimizer.zero_grad()
			# Forward pass
			outputs = model(inputs)
			if type(outputs) == tuple:
				outputs, _ = outputs
			# Calculate loss
			_, preds = torch.max(outputs.data, 1)  # get the index of the max log-probability
			loss = criterion(outputs, labels)
			# Backward + optimize
			loss.backward()
			optimizer.step()
			running_loss += loss.data
			running_corrects += preds.cpu().eq(labels.cpu()).sum()
		epoch_loss = running_loss.item() / len(dataset)
		epoch_acc = running_corrects.item() / float(len(dataset))
		train_logger.add_scalar('Epoch Loss ', epoch_loss, count)
		print('Loss: {:.4f} Acc: {:.4f} lr: {:.2e}\n'.format(epoch_loss, epoch_acc, optimizer.param_groups[0]['lr']))
			
		print("Validating")
		model.eval()
		running_corrects = 0.0
		for inputs, metas in tqdm(test_dataloader):
			labels = Variable(metas).to(device)
			inputs = Variable(inputs).to(device)
			outputs = model(inputs)
			if type(outputs) == tuple:
				outputs, _ = outputs
			_, preds = torch.max(outputs.data, 1)  # get the index of the max log-probability
			running_corrects += preds.cpu().eq(labels.cpu()).sum()  # 
		epoch_acc = running_corrects.item() / float(len(test_dataset))
		validation_logger.add_scalar('Epoch Acc ', running_corrects.item(), count)
		print('Acc: {:.4f}\n'.format(epoch_acc))

		# Deep copy the model
		if epoch_acc >= best_acc:
			best_acc = epoch_acc
			best_model_wts = model.state_dict()
			torch.save(model.state_dict(), "./best_model_resnet.pth")
	# Display stats
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))




if __name__ == '__main__':
	main()