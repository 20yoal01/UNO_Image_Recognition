import cv2 as cv
import numpy as np


def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


blank = np.zeros((500, 500, 3), dtype="uint8")

# 1. Paint the image a certain color

# blank[200:300] = 255, 0, 0
# cv.imshow('Green', blank)

# 2. Draw a Rectangle
# cv.rectangle(blank, (0, 0), (blank.shape[1]//2,
#             blank.shape[0]//2), (0, 255, 0), thickness=cv.FILLED)
# cv.imshow('Rectangle', blank)


# 3. Draw A circle
# cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2),
#          40, (0, 255, 0), thickness=-1)
# cv.imshow("Circle", blank)


# 4. Draw A line
# cv.line(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2),
#        (255, 255, 255), thickness=3)
# cv.imshow('Line', blank)

# 5. Write text on screen
cv.putText(blank, 'Hello World!', (225, 225),
           cv.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 255, 0), thickness=2)
cv.imshow('Text', blank)

# img = cv.imread('Photos/20230406_151337.jpg')
# img_resized = rescaleFrame(img)
# cv.imshow('UNO', img_resized)

cv.waitKey(0)
