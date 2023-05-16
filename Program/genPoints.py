import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random

LOWER_THRESHOLD = 100
UPPER_THRESHOLD = 255
APERTURE_SIZE = 5
UNO_CARD_PATH = 'photos/new/yellow/'
SATURATION_RANGE = (0, 30)
VALUE_RANGE = (0, 40)

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
    img = cv.resize(img, None, fx= 0.5, fy= 0.5, interpolation=cv.INTER_AREA)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 11)
    blur = cv.bilateralFilter(blur, 9, 130, 130)
        
    canny = cv.Canny(blur, LOWER_THRESHOLD, UPPER_THRESHOLD, apertureSize=APERTURE_SIZE)
    karnel = np.ones((5,5), np.uint8)
    canny = cv.dilate(canny, karnel,iterations=1)
    ret, threshold = cv.threshold(canny, 125, 255, cv.THRESH_BINARY)
    contours, hier = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    mask = np.zeros(shape=img.shape[:2], dtype=np.uint8)

    index_sort = sorted(range(len(contours)), key=lambda i : cv.contourArea(contours[i]), reverse=True)

    cnt_sorted = []
    hier_sorted = []
    cards = []

    contour_is_card = np.zeros(len(contours), dtype=int)
    
    for i in index_sort:
        cnt_sorted.append(contours[i])
        hier_sorted.append(hier[0][i])
    
    for i in range(len(contours)):
        size = cv.contourArea(cnt_sorted[i])
        peri = cv.arcLength(cnt_sorted[i],True)
        approx = cv.approxPolyDP(cnt_sorted[i],0.02*peri,True)
        if((size < 120_000) and (size > 0) and (hier_sorted[i][3] == -1) and (len(approx) == 4)):
            contour_is_card[i] = 1

    print(contour_is_card)

    for i in range(len(cnt_sorted)):
        if (contour_is_card[i] == 1):
            cv.drawContours(mask, [cnt_sorted[i]], -1, (255,225,225), -1)
    #img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    #print(img_hsv)
    mask = cv.erode(mask, None, iterations=2)
    cv.imshow('d',mask)
    cv.waitKey(0)
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
np.savetxt('new_yellow_bgr.csv', arr, fmt="%s", delimiter=",")