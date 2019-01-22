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
	parser.add_argument('--dataroot', type=str, default='/home/meghana/Desktop/SKIN/skin-cancer-mnist-ham10000/', help='path to dataset')
	parser.add_argument('--model_path', default='./best_model.pth', help='folder to output images and model checkpoints')
	return parser.parse_args()

def main():
	print("Start!")
	args = parse_args()

	test_dataset = dataset_reader.import_test_dataset(args)
	test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

	#model
	n_class = 7

	model = models.resnet50( pretrained=False )
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, n_class)

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = torch.device("cpu")
		print("Using", device)
	model.to(device)  # Move model to device

	model.load_state_dict(torch.load(args.model_path))

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
	print('Acc: {:.4f}\n'.format(epoch_acc))



if __name__ == '__main__':
	main()