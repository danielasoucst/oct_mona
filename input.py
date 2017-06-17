# coding: utf-8
import os
import cv2
import heapq
import numpy as np
from matplotlib import pyplot as plt
import flatten

def load_image(dir):
    lstImages = []
    for file in os.listdir(dir):
        if file.endswith(".tif"):
            volumeName = os.path.join(dir, file)
            print (volumeName)
            img = cv2.imread(volumeName,0)
            lstImages.append(img)

    return lstImages

def apply_filter(image,metodo):
    print(image.shape)
    if(metodo=='gauss'):
        out = cv2.GaussianBlur(image, (1, 3), 0)
        return out
    return None





