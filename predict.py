import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adamax
import random as python_random
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

model = load_model("C:\\Users\\fusyr\\OneDrive\\Documenti\\Riccardo\\Progetti\\skin-cancer-detection-cnn\\multi class classifier\\Skin Cancer.h5")

def predict_cancer(image_path):
    img = keras_image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    i = 0
    for item in predictions[0]:
        print(f"Prediction for {classes[i]} is {item}")
        i = i + 1

    predicted_class = np.argmax(predictions, axis=1)[0]
    # print("Predicted class index:", predicted_class)

    predicted_label = classes[predicted_class]
    # print("Predicted label:", predicted_label)
    
    confidence = predictions[0][predicted_class]


    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    # return None, None, img
    return predicted_label, confidence, img

image_path = "C:\\Users\\fusyr\\OneDrive\\Documenti\\Riccardo\\Progetti\\skin-cancer-detection-cnn\\multi class classifier\\test.jpg"
predicted_label, confidence, img = predict_cancer(image_path)

print(f"Predicted cancer type: {predicted_label}")
print(f"Confidence: {confidence}")