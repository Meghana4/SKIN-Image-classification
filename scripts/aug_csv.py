import pandas as pd
from random import shuffle
import Augmentor
from shutil import copyfile
import os

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
    if label == 'nv':
        nv_list.append(ham_data.iloc[i, 0])
    if label == 'mel':
        mel_list.append(ham_data.iloc[i, 0])
    if label == 'bkl':
        bkl_list.append(ham_data.iloc[i, 0])
    if label == 'bcc':
        bcc_list.append(ham_data.iloc[i, 0])
    if label == 'akiec':
        akiec_list.append(ham_data.iloc[i, 0])
    if label == 'vasc':
        vase_list.append(ham_data.iloc[i, 0])
    if label == 'df':
        df_list.append(ham_data.iloc[i, 0])

# code to tranfer belongingfiles into separate folder to augment
# # df
# for file in vase_list:
#     source = os.path.join('/home/meghana/Desktop/SKIN/skin-cancer-mnist-ham10000/HAM10000_images_part_1', file + '.jpg')
#     if not os.path.exists(source):
#         source = os.path.join('/home/meghana/Desktop/SKIN/skin-cancer-mnist-ham10000/HAM10000_images_part_2', file + '.jpg')
#     target = "/home/meghana/Desktop/SKIN/train_dir/vasc/" + file + ".jpg"
#     # adding exception handling
#     try:
#         copyfile(source, target)
#     except IOError as e:
#         print("Unable to copy file. %s" % e)
#         exit(1)
#     except:
#         print("Unexpected error:", sys.exc_info())
#         exit(1)

# Augmentation logic
# for i in range(80):
#     p = Augmentor.Pipeline("/home/meghana/Desktop/SKIN/train_dir/vasc/")
#     p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
#     p.flip_left_right(probability=0.5)
#     p.zoom_random(probability=0.5, percentage_area=0.8)
#     p.flip_top_bottom(probability=0.5)
#     p.sample(50)




