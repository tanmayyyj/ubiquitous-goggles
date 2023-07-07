import os
import numpy as np
from config import *
from tqdm import tqdm
import random

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

x_train = np.zeros((len(train_ids), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)
y_train = np.zeros((len(train_ids), IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=bool)
x_test = np.zeros((len(test_ids), IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), dtype = np.uint8)


print("\nWorking on the train image data\n")
for n, id_ in tqdm(enumerate(train_ids), total = len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + "/images/" + id_ + ".png")[:,:,:IMAGE_CHANNEL]
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode="constant", preserve_range=True)
    x_train[n] = img

    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype = bool)
    for mask_file in next(os.walk(path + "/masks/"))[2]:
        mask_ = imread(path + "/masks/" + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMAGE_HEIGHT, IMAGE_WIDTH), mode = "constant", preserve_range = True), axis = -1)
        mask = np.maximum(mask, mask_)

    y_train[n] = mask

print("\nWorking on the test image data\n")
size_test = []
for n, id_ in tqdm(enumerate(test_ids), total = len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + "/images/" + id_ + ".png")[:,:,:IMAGE_CHANNEL] #3 Channels from here
    size_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode = "constant", preserve_range = True)
    x_test[n] = img

