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

print(img_resized.shape[:2])

b,g,r = cv.split(img_resized)

blue = cv.merge([b,blank,blank])
green = cv.merge([blank,g,blank])
red = cv.merge([blank,blank,r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img_resized.shape)
print(b.shape)
print(g.shape)
print(r.shape)

merged = cv.merge([b,g,r])
cv.imshow('Merged image', merged)

cv.waitKey(0)