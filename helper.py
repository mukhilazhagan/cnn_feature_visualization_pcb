import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import cv2


def imread(path_str):
    try:
        img = cv2.imread(path_str)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img
    except:
        print("Unable to read source")

def imshow(img):
    plt.figure(figsize=(10, 7)) # Note : figsize is in inches (10,7) works pretty well
    plt.imshow(img)
    plt.show()

img = imread("./database/orig_blue.jpg")
#imshow(img)






def cornerdetect(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 100000, 0.01, 1)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    imshow(img)

cornerdetect(img)
#def color_remove(img, rl, ru, gl, gu, bl, bu):

# green = cv2.inRange(img, (0,0,0), (100, 255, 100))
# notgreen = cv2.bitwise_not(img, img, mask=green)
# imshow(notgreen)