import os
import splitfolders
from glob import glob
import yaml
from keras.preprocessing.image import ImageDataGenerator

# splitfolders.ratio("./face-recognition-dataset", output="./face-recognition-dataset-splitted", seed=1, ratio=(.8, .2))

# train_path = './face-recognition-dataset-splitted/train'
# val_path = './face-recognition-dataset-splitted/val'

# folders = glob(train_path + '/*')
# assert list(set(os.listdir('./face-recognition-dataset-splitted/train'))) == list(set(os.listdir('./face-recognition-dataset-splitted/val')))
# IMAGE_SIZE = [160, 160] 

# train_gen = ImageDataGenerator(
#     rescale=1./255,              
#     shear_range=0.2,           
#     zoom_range=0.2,              
#     horizontal_flip=True         
# )

# val_gen = ImageDataGenerator(
#     rescale=1./255               # Rescale pixel values to be between 0 and 1 (normalization)
# )

# training_set = train_gen.flow_from_directory(
#     train_path,
#     target_size = IMAGE_SIZE,
#     batch_size = 32,
#     class_mode = 'categorical' 
# )

# val_set = val_gen.flow_from_directory(
#     val_path,
#     target_size = IMAGE_SIZE,
#     batch_size = 32,
#     class_mode = 'categorical',
#     shuffle = False,
# )

# class_labels = list(val_set.class_indices.keys())
# with open('class_labels.yaml', 'w') as file:
#     yaml.dump(class_labels, file)

# load classes from yaml file
with open('class_labels.yaml', 'r') as file:
    class_labels = yaml.safe_load(file)

print(class_labels)