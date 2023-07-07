from config import*
from model import *
import matplotlib.pyplot as plt
import numpy as np

def pred_function():
    path = str(input("Enter the path for the image: \n"))

    #Need to run the preprocessing
    img = imread(path)[:,:,:IMAGE_CHANNEL]
    imshow(img)
    plt.show()
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), preserve_range=True, mode="constant")
    img = tf.expand_dims(img, axis = 0)
    model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    model.load_weights("model.h5")

    predictions = model.predict(img) #My metric was accuracy
    imshow(np.squeeze(predictions))
    plt.show()

if __name__ == "__main__":
    pred_function()
