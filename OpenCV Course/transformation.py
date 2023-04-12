import cv2 as cv
import numpy as np


def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Translation


def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x --> Left
# -y --> Up
# x  --> Right
# y  --> Down


# Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 0.8)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)


# Resizing

# Enlaring image: INTER_LINEAR, INTER_CUBIC (CUBIC IS SLOWER BUT BETTER)
# Shrinking image: INTER_AREA

img = cv.imread('OpenCV Course/Photos/20230406_151337.jpg')
img_resized = rescaleFrame(img)
translated = translate(img_resized, 100, 100)
rotated = rotate(img_resized, -45)
flip = cv.flip(img_resized, -1)
cropped = img_resized[200:400, 300:400]

cv.imshow('Original', img_resized)
cv.imshow('Translated', translated)
# cv.imshow('Rotation', rotated)
# cv.imshow('Flipped', flip)
#cv.imshow('Cropped', cropped)

cv.waitKey(0)
