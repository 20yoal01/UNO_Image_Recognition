import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('Photos/20230406_151337.jpg')
img_resized = rescaleFrame(img) 

cv.imshow('UNO', img_resized)

blank = np.zeros(img_resized.shape[:2], dtype='uint8')
cv.imshow('Blank', blank)

circle = cv.circle(blank.copy(), (img_resized.shape[1]//2 + 45, img_resized.shape[0]//2), 100, 255, -1)

rectangle = cv.rectangle(blank.copy(), (150,150), (310,310), 255, -1)
cv.imshow('Rectangle', rectangle)

weird_shape = cv.bitwise_and(circle, rectangle)
cv.imshow('Weird Shape', weird_shape)

masked = cv.bitwise_and(img_resized, img_resized, mask=weird_shape)
cv.imshow('Masked Image', masked)

cv.waitKey(0)