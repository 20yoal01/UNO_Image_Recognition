import cv2 as cv
import numpy as np

def resize(img, scale_percent=.5):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

img = resize(cv.imread(r'frame.jpg'))
#cv.imshow('UNO', img)

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(grey, (9,9), cv.BORDER_DEFAULT)
blur2 = cv.medianBlur(grey, 11)
blur3 = cv.bilateralFilter(blur2,9,130,130)

t_lower = 100  # Lower Threshold
t_upper = 255  # Upper threshold
aperture_size = 5  # Aperture size

canny = cv.Canny(blur3, t_lower, t_upper, apertureSize=aperture_size)

#cv.imshow('canny', canny)

# Dilate the edges to close any gaps in the white outline
kernel = np.ones((5,5),np.uint8)
canny = cv.dilate(canny,kernel,iterations = 1)

ret, thresh = cv.threshold(canny, 125, 255, cv.THRESH_BINARY)
contours, hierarchies = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 

mask = np.zeros(shape=img.shape, dtype='uint8')
cv.drawContours(mask, contours, -1, (255,255,255), thickness=cv.FILLED)

#cv.imshow('A', img)

#cv.imshow('A', mask)

#blank = np.zeros(shape=img.shape, dtype='uint8')
#blank[:] = (0, 0, 255)
blank = cv.imread('namn.jpg')

cv.copyTo(img, mask, blank)
cv.normalize(mask.copy(), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

cv.imshow('Blank', blank)
y,x,c = img.shape
print("Bildens storlek är: ", x, " x ", y)
#cv.imshow('Contours', mask)
cv.waitKey(0)