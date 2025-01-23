import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.utils import to_categorical
import random as python_random
from keras.callbacks import ReduceLROnPlateau
tf.random.set_seed(42)
np.random.seed(42)
python_random.seed(42)

IMAGE_SIZE = 28
classes = {4: ('nv', ' melanocytic nevi'),
           6: ('mel', 'melanoma'),
           2 :('bkl', 'benign keratosis-like lesions'), 
           1:('bcc' , ' basal cell carcinoma'),
           5: ('vasc', ' pyogenic granulomas and hemorrhage'),
           0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
           3: ('df', 'dermatofibroma')}

model = load_model("C:\\Users\\fusyr\\OneDrive\\Documenti\\Riccardo\\Progetti\\skin_cancer_detection\\Skin Cancer.h5")
num_classes = len(classes)  # Total number of classes

def preprocess_images(image_paths, labels):
    """
    Preprocess a list of image paths and their corresponding labels.
    
    Args:
        image_paths (list): List of file paths to images.
        labels (list): List of corresponding labels for each image.
        
    Returns:
        img_array (numpy.ndarray): Array of preprocessed images.
        y (numpy.ndarray): One-hot encoded labels.
    """
    img_array = []
    for path in image_paths:
        img = keras_image.load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img = keras_image.img_to_array(img)
        img_array.append(img)
    
    img_array = np.array(img_array).reshape(-1, 28, 28, 3)  # Convert list to NumPy array
    
    # One-hot encode labels
    y = to_categorical(labels, num_classes)
    
    return img_array, y

# Example usage
image_paths = [
    "C:\\path_to_image1.jpg",
    "C:\\path_to_image2.jpg",
    "C:\\path_to_image3.jpg",
]
labels = [2, 6, 4]

img_array, y = preprocess_images(image_paths, labels)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy'
                                            , patience = 2
                                            , verbose=1
                                            ,factor=0.5
                                            , min_lr=0.00001)

model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

model.fit(img_array ,
          y ,
          epochs=25 ,
          batch_size=128,
          callbacks=[learning_rate_reduction])

model.save('Skin Cancer 2.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model) 
tflite_model = converter.convert() 

print("model converted")

# Save the model. 
with open('Skin.tflite', 'wb') as f:
    f.write(tflite_model)