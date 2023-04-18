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
cv.imshow('thresh', thresh)
contours, hierarchies = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

index_sort = sorted(range(len(contours)), key=lambda i : cv.contourArea(contours[i]),reverse=True)

contour_sort = []
hierarchie_sort = []

mask = np.zeros(shape=img.shape, dtype='uint8')

for i in index_sort:
    contour_sort.append(contours[i])
    hierarchie_sort.append(hierarchies[i])

for i in range(len(contour_sort)):
    size = cv.contourArea(contour_sort[i])
    peri = cv.arcLength(contour_sort[i],True)
    approx = cv.approxPolyDP(contour_sort[i],0.01*peri,True)

    if((size < 3000000) and (size > 20) and (hierarchie_sort[i][3] == -1) and (len(approx) == 4)):
        mask[i] = 1


cv.drawContours(mask, contours, -1, (255,255,255), thickness=cv.FILLED)

#cv.imshow('A', img)

#cv.imshow('A', mask)

blank = np.zeros(shape=img.shape, dtype='uint8')
blank[:] = (0, 0, 0)
#blank = cv.imread('namn.jpg')

cv.copyTo(img, mask, blank)
cv.normalize(mask.copy(), None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)

cv.imshow('Blank', blank)
#cv.imshow('Contours', mask)
cv.waitKey(0)