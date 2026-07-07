import tensorflow as tf
import numpy as np
import json
import cv2

from config import MODEL_PATH, CLASS_NAMES, IMAGE_SIZE


# Load model

model = tf.keras.models.load_model(MODEL_PATH)


# Load class names

with open(CLASS_NAMES,"r") as file:
    class_names=json.load(file)



def predict_image(image_path):

    img=cv2.imread(image_path)

    img=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    img=cv2.resize(
        img,
        IMAGE_SIZE
    )


    img=img/255.0


    img=np.expand_dims(
        img,
        axis=0
    )


    prediction=model.predict(img)


    index=np.argmax(prediction)


    confidence=float(
        np.max(prediction)*100
    )


    disease=class_names[index]


    return disease,confidence
