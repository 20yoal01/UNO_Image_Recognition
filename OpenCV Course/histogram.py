import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('OpenCV Course/Photos/20230406_151337.jpg')
img_resized = rescaleFrame(img) 

cv.imshow('UNO', img_resized)

blank = np.zeros(img_resized.shape[:2], dtype='uint8')

""" gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray) """

mask = cv.circle(blank, (img_resized.shape[1]//2, img_resized.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(img_resized, img_resized, mask=mask)
cv.imshow('Mask', masked) 

# Grayscale histogram
#gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256])

""" plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel(' # of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show() """

# Colour Histogram 

plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel(' # of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img_resized], [i], mask, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()

cv.waitKey(0)