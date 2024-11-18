import PIL
import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image , ImageOps

base = 'D:\projects\Data Science\BrainTumorDetection\model'
model = keras.models.load_model(f'{base}/model.h5')

def image_pre(path):
    print(path)
    data = np.ndarray(shape=(1,150, 150, 1), dtype=np.float32)
    size = (150, 150)
    image = Image.open(path)
    image = ImageOps.grayscale(image)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    data = image_array.reshape((-1,150,150,1))
    return data

def predict(data):
    prediction = model.predict(data)
    return prediction[0][0]
