# SKIN-Image-classification

Description:
The task is perform image classification to lesion diagnosis.There are 7 classes in the HAM10000 Dataset.The number of images in each category is extremely imbalanced. For example, the largest category, Melanocytic nevus, has 6705 images
while the smallest category, dermatofibroma, only has 115 images.

Method:
Experiment is performed by training the multi-class classification model on ResNet50. 
The dataset is split into 80% training and 20% validation set. The fully connected layer in base model is modified according to number of classes required.
Softmax is used as the activation function in the prediction layer and multi-class cross entropy as the loss function.
To handle the imbalance in dataset, i augment the images of classes expect 'nv' using imaugmenter. Refer https://github.com/mdbloice/Augmentor for more details.

Results:
Able to achieve 77.61% accuracy on test set containing 2005 images containing all classes.

# Data preprocessing
Use create_train_test_csv.py in scripts folder to split data into training and test set. It creates train.csv and test.csv
Use Augmenter to augment the classes except 'nv'. Script aug_csv.py can be used as reference.
After augmentation create CSV file for both train and test datasets containing absolute paths of files and class label.
sample csv files can be seen in scripts folder.

# To train the model
python main.py --dataroot [path to folder containing train and test csv files] --epochs 20 --batch_size 32

# To test the model
python test.py --dataroot [path to folder containing train and test csv files] --model_path [path to model to be tested]
