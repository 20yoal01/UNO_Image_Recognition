import cv2 as cv
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('simpsons.model')
test_path = r'C:\Users\eyoalxa\Documents\Python OpenCV\simpsons_recognition\input\simpsons_dataset\simpsons_dataset\bart_simpson\pic_0004.jpg'

img = cv.imread(test_path)

def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    return img

predictions = model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])