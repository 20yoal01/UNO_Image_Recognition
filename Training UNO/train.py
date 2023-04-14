import cv2 as cv
import numpy as np

PATH = 'UNO Syn/UNO.jpg'

def resize(img, scale_percent=.05):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

img = resize(cv.imread(r'C:\Users\eyoalxa\Documents\Python OpenCV\OpenCV Course\Photos\20230406_152117.jpg'))
#cv.imshow('UNO', img)

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(grey, (7,7), cv.BORDER_DEFAULT)

t_lower = 70  # Lower Threshold
t_upper = 255  # Upper threshold
aperture_size = 5  # Aperture size

canny = cv.Canny(blur, t_lower, t_upper, apertureSize=aperture_size)

cv.imshow('canny', canny)

# Dilate the edges to close any gaps in the white outline
kernel = np.ones((5,5),np.uint8)
canny = cv.dilate(canny,kernel,iterations = 1)

ret, thresh = cv.threshold(canny, 125, 255, cv.THRESH_BINARY)
contours, hierarchies = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
#cv.imshow('Canny', canny)

mask = np.zeros(shape=img.shape, dtype='uint8')

cv.drawContours(mask, contours, -1, (255,255,255), thickness=cv.FILLED)

#cv.imshow('A', img)

#cv.imshow('A', mask)

blank = np.zeros(shape=img.shape, dtype='uint8')
blank[:] = (0, 0, 0)

cv.copyTo(img, mask, blank)
cv.normalize(mask.copy(), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

cv.imshow('Blank', blank)
cv.imshow('Contours', mask)
cv.imwrite(PATH, blank)
cv.waitKey(0)