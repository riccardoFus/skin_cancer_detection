import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import tensorflow as tf
import os
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

with open("C:\\Users\\fusyr\\OneDrive\\Documenti\\Riccardo\\Progetti\\skin-cancer-detection-cnn\\skin-cancer-prediction\\model\\class_indices.json", 'r') as json_file:
    class_indices = json.load(json_file)

def predict_cancer(image_path):
    img = keras_image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = keras_image.img_to_array(img)  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    for i, item in enumerate(predictions[0]):
        print(f"Prediction for {classes[i]} is {item}")

    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = classes[predicted_class]
    confidence = predictions[0][predicted_class]

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    return predicted_label, confidence, img

def predict_on_directory(directory_path):
    results = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):  # Ensure it's a file
            try:
                predicted_label, confidence, _ = predict_cancer(file_path)
                results.append((filename, predicted_label, confidence))
                print(f"Image: {filename}, Predicted: {predicted_label}, Confidence: {confidence}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return results

# Directory containing test images
test_dir = "C:\\Users\\fusyr\\OneDrive\\Documenti\\Riccardo\\Progetti\\skin-cancer-detection-cnn\\multi class classifier\\test_photo\\Test\\basal cell carcinoma"
results = predict_on_directory(test_dir)

# Print final results
print("\nSummary of Predictions:")
for filename, label, confidence in results:
    print(f"Image: {filename}, Predicted: {label}, Confidence: {confidence}")
