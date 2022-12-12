import cv2
import torch
from PIL import Image
import numpy as np

def getMap(img, map_thresh):
    map_thresh = 1-map_thresh
    # Convert PIL to cv2
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # Convert to Gray Scale Image
    src0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gray inversion 
    src = cv2.bitwise_not(src0)
    # Threshold truncation
    hist = cv2.calcHist([src], [0], None, [256], [0, 256])
    reversed_hist = reversed(hist)
    n_pix = src.size
    cut_n_pix = int(n_pix*map_thresh)  # map_thresh before brightness

    # Calculate the pixel threshold thresh
    temp = 0
    for val, val_n_pix in enumerate(reversed_hist):
        temp += val_n_pix
        if temp >= cut_n_pix:
            thresh = val
            break

    _, map = cv2.threshold(src, thresh, 255, cv2.THRESH_TOZERO)

    # Covert cv2 back to PIL
    map = Image.fromarray(map)
    
    return map
