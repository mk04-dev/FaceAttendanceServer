"""
# This script is used to train a ResNet50 model for face recognition using a dataset of images.
# It splits the dataset into training and validation sets, preprocesses the images, and trains the model.
# The model is then evaluated on the validation set, and various metrics such as accuracy, precision, recall, and F1-score are calculated.
# The script also generates and saves plots for training loss, accuracy, and confusion matrix.
#
# The model is saved to a file for later use.
# NOTE: use capture_face.py to collect images for training
# The dataset should be organized in the following structure:
# face-recognition-dataset/
# ├── empId1
# │   ├── empId1_1.jpg
# │   ├── empId1_2.jpg
# │   └── ...
# ├── empId2
# │   ├── empId2_1.jpg
# │   ├── empId2_2.jpg
# │   └── ...
"""

import os
import warnings
import splitfolders
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import yaml
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score
warnings.simplefilter('ignore')
pd.options.display.float_format = '{:.2f}'.format

tf.__version__

# Check if GPU is available and set it for TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Enable memory growth for the GPU
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"GPU is available and configured: {physical_devices}")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found. Using CPU.")

splitfolders.ratio("./face-recognition-dataset", output="./face-recognition-dataset-splitted", seed=1, ratio=(.8, .2))

train_path = './face-recognition-dataset-splitted/train'
val_path = './face-recognition-dataset-splitted/val'

folders = glob(train_path + '/*')
len(folders)

assert list(set(os.listdir('./face-recognition-dataset-splitted/train'))) == list(set(os.listdir('./face-recognition-dataset-splitted/val')))

# Reset Image size
IMAGE_SIZE = [160, 160] 

# Load Pre-Trained Model
resnet = ResNet50(
    input_shape = IMAGE_SIZE + [3],
    weights = 'imagenet', #Default
    include_top = False  
)

# Using default weights used by imagenet
for layer in resnet.layers:
    layer.trainable = False

# Flatten the output of the ResNet model to convert the 2D feature maps to 1D feature vectors.
x = Flatten()(resnet.output)

# The 'softmax' activation function is used to produce a probability distribution over the classes.
prediction = Dense(len(folders), activation='softmax')(x)

# The model takes the input from the ResNet model and outputs the predictions from the dense layer.
model = Model(inputs=resnet.input, outputs=prediction)

# # Print a summary of the model architecture.
# model.summary()

model.compile(
    loss='categorical_crossentropy',  # Loss function used for multi-class classification tasks
    optimizer='adam',                # Optimizer that adjusts the weights of the neural network
    metrics=['accuracy']            
)

"""
-Rescale pixel values to be between 0 and 1 (normalization)
-Apply random shearing transformations to the images
-Apply random zoom transformations to the images
-Randomly flip images horizontally
"""

train_gen = ImageDataGenerator(
    rescale=1./255,              
    shear_range=0.2,           
    zoom_range=0.2,              
    horizontal_flip=True         
)

val_gen = ImageDataGenerator(
    rescale=1./255               # Rescale pixel values to be between 0 and 1 (normalization)
)

training_set = train_gen.flow_from_directory(
    train_path,
    target_size = IMAGE_SIZE,
    batch_size = 32,
    class_mode = 'categorical' 
)

val_set = val_gen.flow_from_directory(
    val_path,
    target_size = IMAGE_SIZE,
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = False,
)

history = model.fit(
    training_set,
    validation_data=val_set,
    epochs=100
)

model.save('./model.h5')

# Plotting the training loss
plt.plot(history.history['loss'], label='train_loss')

# Plotting the validation loss
plt.plot(history.history['val_loss'], label='val loss')

# Adding a legend to distinguish between the training and validation loss plots
plt.legend()

# Save the plot as an image file
plt.savefig('./loss_plot.png')

# Plot the Accuracy
plt.plot(history.history['val_accuracy'], label ='val accuracy')
plt.plot(history.history['accuracy'], label = 'train accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('./accuracy_plot.png')

prediction = model.predict(val_set)
prediction = np.argmax(prediction, axis = 1)

true_classes = val_set.classes

accuracy = accuracy_score(true_classes, prediction)
print(f"Accuracy: {accuracy * 100:.2f}%")

class_labels = list(val_set.class_indices.keys())
report = classification_report(true_classes, prediction, target_names = class_labels, output_dict = True)
# save class_labels to yaml file
with open('class_labels.yaml', 'w') as file:
    yaml.dump(class_labels, file)


report_df = pd.DataFrame(report).transpose()
report_df.to_csv('./classification_report.csv', index = True)

conf_matrix = confusion_matrix(true_classes, prediction)

cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = class_labels)

cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('./confusion_matrix.png')

#Plot Precision, Recall and Precision Score
metrics_data = report_df.loc[class_labels, ['precision', 'recall', 'f1-score']]
accuracies = []

for label in class_labels:
    idx = val_set.class_indices[label]
    tp = conf_matrix[idx, idx]
    total = np.sum(conf_matrix[idx, :])
    accuracies.append(tp/total)
    
metrics_data['accuracy'] = accuracies

metrics_data.plot(kind='bar', figsize = (8, 4))

plt.title('Precison, Recall, F1-Score and Accuracy for Each Class\n')
plt.xlabel('Class')
plt.ylabel('Score')
plt.ylim(0,1.1)
plt.legend(loc = 'lower right')
plt.xticks(rotation = 0)
plt.grid(axis = 'y')
plt.savefig('./metrics_plot.png')