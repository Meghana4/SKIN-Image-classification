import pandas as pd
from random import shuffle
import os
import sys
from glob import glob
import os, math, random
from os.path import *
import numpy as np

ham_data = pd.read_csv('train.csv')

df_list = []
nv_list = []
bkl_list = []
bcc_list = []
akiec_list = []
vase_list = []
mel_list = []

for i in range(len(ham_data)):
    label = ham_data.iloc[i, 1]
    source = os.path.join('/home/meghana/Desktop/SKIN/skin-cancer-mnist-ham10000/HAM10000_images_part_1', ham_data.iloc[i, 0] + '.jpg')
    if not os.path.exists(source):
        source = os.path.join('/home/meghana/Desktop/SKIN/skin-cancer-mnist-ham10000/HAM10000_images_part_2', ham_data.iloc[i, 0] + '.jpg')
    if label == 'nv':
        nv_list.append(source)
    if label == 'mel':
        mel_list.append(source)
    if label == 'bkl':
        bkl_list.append(source)
    if label == 'bcc':
        bcc_list.append(source)
    if label == 'akiec':
        akiec_list.append(source)
    if label == 'vasc':
        vase_list.append(source)
    if label == 'df':
        df_list.append(source)

print("nv")
print(len(nv_list))
print("mel")
print(len(mel_list))
print("bkl")
print(len(bkl_list))
print("bcc")
print(len(bcc_list))
print("akiec")
print(len(akiec_list))
print("vasc")
print(len(vase_list))
print("df")
print(len(df_list))

train_list_images = []
train_list_labels = []

for item in mel_list:
    train_list_images.append(item)
    train_list_labels.append('mel')

for item in bkl_list:
    train_list_images.append(item)
    train_list_labels.append('bkl')

for item in bcc_list:
    train_list_images.append(item)
    train_list_labels.append('bcc')

count = 0
for item in nv_list:
    # if (count > 200):
    #     continue
    # count += 1
    train_list_images.append(item)
    train_list_labels.append('nv')

for item in akiec_list:
    train_list_images.append(item)
    train_list_labels.append('akiec')

for item in vase_list:
    train_list_images.append(item)
    train_list_labels.append('vasc')

for item in df_list:
    train_list_images.append(item)
    train_list_labels.append('df')

print(len(train_list_images))

train_dataframe = pd.DataFrame(
    {
        "Images":train_list_images,
        "Labels":train_list_labels
    }
)

train_dataframe.to_csv("test_new.csv", sep='\t')
