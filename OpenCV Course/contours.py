import cv2 as cv
import numpy as np

def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread('OpenCV Course/Photos/20230406_151337.jpg')
img_resized = rescaleFrame(img)

blank = np.zeros(img_resized.shape, dtype='uint8')
cv.imshow('Blank', blank)

grey = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

#cv.imshow('Grey', grey)
#cv.imshow('Original', img_resized)

blur = cv.GaussianBlur(grey, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny', canny)

ret, thresh = cv.threshold(canny, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)

contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE) 
print(f'{len(contours)} contours(s) found')

cv.drawContours(blank, contours, -1, (0,0,255), thickness=1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)