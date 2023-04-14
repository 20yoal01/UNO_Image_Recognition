import cv2 as cv
import numpy as np

PATH = 'UNO Syn/UNO.jpg'

def resize(img, scale_percent=.05):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

img = resize(cv.imread(r'C:\Users\eyoalxa\Documents\Python OpenCV\OpenCV Course\Photos\20230406_152117.jpg'))

grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(grey, (5,5), cv.BORDER_DEFAULT)

t_lower = 50  # Lower Threshold
t_upper = 200  # Upper threshold
aperture_size = 5  # Aperture size

canny = cv.Canny(blur, 50, 200, apertureSize=aperture_size)

# Dilate the edges to close any gaps in the white outline
kernel = np.ones((5,5),np.uint8)
canny = cv.dilate(canny,kernel,iterations = 1)

ret, thresh = cv.threshold(canny, 125, 255, cv.THRESH_BINARY)
contours, hierarchies = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 

cv.imshow('th', thresh)

# Find the longest contour
max_contour = None
max_length = 0
for contour in contours:
    length = cv.arcLength(contour, True)
    if length > max_length:
        max_length = length
        max_contour = contour

# Draw the longest contour on the mask
mask = np.zeros(shape=img.shape[:2], dtype='uint8')
cv.drawContours(mask, [max_contour], -1, (255,255,255), thickness=cv.FILLED)

# Copy the original image onto a blank image using the mask
output = np.zeros(shape=img.shape, dtype='uint8')
cv.copyTo(img, mask, output)

# Show the output image
cv.imshow('Output', output)
cv.imwrite(PATH, output)
cv.waitKey(0)