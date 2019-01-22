import torch
import torch.utils.data as data
import os, math, random
from os.path import *
import numpy as np
from glob import glob
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

class HAMdataset(data.Dataset):
    def __init__(self, args,file=None,flag=1):
        self.args = args
        self.dict = {
            'nv': 0,
            'mel': 1,
            'bkl': 2,
            'bcc': 3,
            'akiec': 4,
            'vasc': 5,
            'df':  6,
        }
        self.flag = flag
        self.ham_data = pd.read_csv(os.path.join(self.args.dataroot, file))
        self.length = 10015
        self.transform = transform=transforms.Compose([
            transforms.Resize(224),  # required size
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # required normalisation
        ])


    def __getitem__(self, index):
        # # Collect image name from csv frame
        # img_name = os.path.join(self.args.dataroot, 'HAM10000_images_part_1', self.ham_data.iloc[index, 0] + '.jpg')
        # if not os.path.exists(img_name):
        #     img_name = os.path.join(self.args.dataroot, 'HAM10000_images_part_2', self.ham_data.iloc[index, 0] + '.jpg')

        # Load image
        image = Image.open(self.ham_data.iloc[index, 0]).convert('RGB')
        # if (self.flag == 1):
        image = self.transform(image)

        # Load meta
        meta = self.ham_data.iloc[index, 1]
        label = self.dict[meta]
        # print(label)

        return image, label

    def __len__(self):
        return len(self.ham_data)

def import_train_dataset(args):
    return HAMdataset(args,file="train_aug.csv",flag=1)

def import_test_dataset(args):
    return HAMdataset(args,file="test_new.csv",flag=0)