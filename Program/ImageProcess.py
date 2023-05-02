import cv2 as cv
import numpy as np

SCALE_PERCENT=.05

class Process:

    def start(img):
        img = cv.resize(img, None, fx= SCALE_PERCENT, fy= SCALE_PERCENT, interpolation=cv.INTER_AREA)
        

        return img


    