import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random

LOWER_THRESHOLD = 0
UPPER_THRESHOLD = 255
APERTURE_SIZE = 3
UNO_TYPE = 'yellow'
UNO_CARD_PATH = 'photos/' + UNO_TYPE + '/'
SATURATION_RANGE = (0, 100)
VALUE_RANGE = (-30, 255)

def changeHSV(img, saturation, value):
    changeImg = img.copy()
    changeImg = cv.cvtColor(changeImg, cv.COLOR_BGR2HSV).astype("float32")
    h,s,v = cv.split(changeImg)
    lim = 255 - value 
    v[v > lim] = 255 
    v[v <= lim] += value
    lim = 255 - saturation 
    s[s > lim] = 255 
    s[s <= lim] += saturation
    changeImg = cv.merge([h,s,v])
    changeImg = cv.cvtColor(changeImg.astype("uint8"), cv.COLOR_HSV2BGR)
    return changeImg

file_dir = os.listdir(UNO_CARD_PATH)
loop_range = int(150/len(file_dir))
color_array = []

for file in file_dir:
    file_path = os.path.join(UNO_CARD_PATH,file)
    img = cv.imread(file_path)
    img = cv.resize(img, None, fx= 0.03, fy= 0.03, interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(img_gray, (7,7), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, LOWER_THRESHOLD, UPPER_THRESHOLD, apertureSize=APERTURE_SIZE)

    kernel = np.ones((5,5),np.uint8)
    canny = cv.dilate(canny,kernel,iterations = 1)
    contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 

    mask = np.zeros(img.shape[:2], dtype='uint8')
    cv.drawContours(mask, contours, -1, (255,255,255), -1)
    mask = cv.erode(mask, None, iterations=2)
    blank = np.ones(img.shape[:2], dtype='uint8')
    #img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    #print(img_hsv)
    mean_color = cv.mean(img, mask=mask)[:3]
    color_array.append([str(int(mean_color[0])),str(int(mean_color[1])),str(int(mean_color[2]))])

    for x in range(loop_range):
        saturation = round(random.uniform(SATURATION_RANGE[0], SATURATION_RANGE[1]), 2)
        value = round(random.uniform(VALUE_RANGE[0],VALUE_RANGE[1]), 2)
        img_hsv_mod = changeHSV(img,saturation,value)
        #img_hsv_mod = cv.cvtColor(img_hsv_mod, cv.COLOR_HSV2RGB)
        mean_color = cv.mean(img_hsv_mod, mask=mask)[:3]
        color_array.append([str(int(mean_color[0])),str(int(mean_color[1])),str(int(mean_color[2]))])

arr = np.array(color_array)
np.savetxt('new_in/' + UNO_TYPE + '_bgr.csv', arr, fmt="%s", delimiter=",")