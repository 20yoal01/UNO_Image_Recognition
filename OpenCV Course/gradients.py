import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('OpenCV Course/Photos/20230406_151337.jpg')
img_resized = rescaleFrame(img) 

#cv.imshow('UNO', img_resized)

gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

#cv.imshow('Sobel X', sobelx)
#cv.imshow('Sobel Y', sobely)
cv.imshow('Combed Sobel', combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)