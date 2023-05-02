import cv2 as cv
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist

LOWER_THRESHOLD = 0
UPPER_THRESHOLD = 255
APERTURE_SIZE = 3

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)
yellow = (255,255,0)
black = (0,0,0)

img = cv.imread(r'C:\Users\ejestxa\Documents\img\UNO_Image_Recognition\photos\20230414_143429.jpg')
width = int(img.shape[1] * 0.05)
height = int(img.shape[0] * 0.05)
dim = (width, height)
img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(img_gray, (7,7), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, LOWER_THRESHOLD, UPPER_THRESHOLD, apertureSize=APERTURE_SIZE)

kernel = np.ones((5,5),np.uint8)
canny = cv.dilate(canny,kernel,iterations = 1)
contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 

mask = np.zeros(img.shape[:2], dtype='uint8')
cv.drawContours(mask, contours, -1, (255,255,255), -1)
mask = cv.erode(mask,None, iterations=2)
mask = cv.mean(img, mask=mask)[:3]
print(mask) #färg i bgr (eventuellt)
