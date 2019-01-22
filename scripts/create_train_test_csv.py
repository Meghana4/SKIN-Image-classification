import pandas as pd
from random import shuffle

ham_data = pd.read_csv('HAM10000_metadata.csv')

df_list = []
nv_list = []
bkl_list = []
bcc_list = []
akiec_list = []
vase_list = []
mel_list = []

for i in range(len(ham_data)):
    label = ham_data.iloc[i, 2]
    if label == 'nv':
        nv_list.append(ham_data.iloc[i, 1])
    if label == 'mel':
        mel_list.append(ham_data.iloc[i, 1])
    if label == 'bkl':
        bkl_list.append(ham_data.iloc[i, 1])
    if label == 'bcc':
        bcc_list.append(ham_data.iloc[i, 1])
    if label == 'akiec':
        akiec_list.append(ham_data.iloc[i, 1])
    if label == 'vasc':
        vase_list.append(ham_data.iloc[i, 1])
    if label == 'df':
        df_list.append(ham_data.iloc[i, 1])

# print("nv")
# print(len(nv_list))
# print("mel")
# print(len(mel_list))
# print("bkl")
# print(len(bkl_list))
# print("bcc")
# print(len(bcc_list))
# print("akiec")
# print(len(akiec_list))
# print("vasc")
# print(len(vase_list))
# print("df")
# print(len(df_list))

shuffle(nv_list)
shuffle(mel_list)
shuffle(bkl_list)
shuffle(bcc_list)
shuffle(akiec_list)
shuffle(vase_list)
shuffle(df_list)

index = int(len(nv_list)*0.8)
train_nv_list = nv_list[:index]
test_nv_list = nv_list[index:]

index = int(len(mel_list)*0.8)
train_mel_list = mel_list[:index]
test_mel_list = mel_list[index:]

index = int(len(bkl_list)*0.8)
train_bkl_list = bkl_list[:index]
test_bkl_list = bkl_list[index:]

index = int(len(bcc_list)*0.8)
train_bcc_list = bcc_list[:index]
test_bcc_list = bcc_list[index:]

index = int(len(akiec_list)*0.8)
train_akiec_list = akiec_list[:index]
test_akiec_list = akiec_list[index:]

index = int(len(vase_list)*0.8)
train_vase_list = vase_list[:index]
test_vase_list = vase_list[index:]

index = int(len(df_list)*0.8)
train_df_list = df_list[:index]
test_df_list = df_list[index:]

train_list_images = []
train_list_labels = []
test_list_images = []
test_list_labels = []

for item in train_nv_list:
    train_list_images.append(item)
    train_list_labels.append('nv')
for item in test_nv_list:
    test_list_images.append(item)
    test_list_labels.append('nv')

for item in train_mel_list:
    train_list_images.append(item)
    train_list_labels.append('mel')
for item in test_mel_list:
    test_list_images.append(item)
    test_list_labels.append('mel')

for item in train_bkl_list:
    train_list_images.append(item)
    train_list_labels.append('bkl')
for item in test_bkl_list:
    test_list_images.append(item)
    test_list_labels.append('bkl')

for item in train_bcc_list:
    train_list_images.append(item)
    train_list_labels.append('bcc')
for item in test_bcc_list:
    test_list_images.append(item)
    test_list_labels.append('bcc')

for item in train_akiec_list:
    train_list_images.append(item)
    train_list_labels.append('akiec')
for item in test_akiec_list:
    test_list_images.append(item)
    test_list_labels.append('akiec')

for item in train_vase_list:
    train_list_images.append(item)
    train_list_labels.append('vasc')
for item in test_vase_list:
    test_list_images.append(item)
    test_list_labels.append('vasc')

for item in train_df_list:
    train_list_images.append(item)
    train_list_labels.append('df')
for item in test_df_list:
    test_list_images.append(item)
    test_list_labels.append('df')

print(len(train_list_images))
print(len(test_list_images))

train_dataframe = pd.DataFrame(
    {
        "Images":train_list_images,
        "Labels":train_list_labels
    }
)

test_dataframe = pd.DataFrame(
    {
        "Images":test_list_images,
        "Labels":test_list_labels
    }
)

train_dataframe.to_csv("train.csv", sep='\t')
test_dataframe.to_csv("test.csv", sep='\t')

