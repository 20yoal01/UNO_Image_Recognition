import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random

LOWER_THRESHOLD = 100
UPPER_THRESHOLD = 200
APERTURE_SIZE = 3
UNO_TYPE = ['BLUE', 'GREEN', 'RED', 'YELLOW']
#UNO_CARD_PATH = 'Training UNO/Photos V2/' + UNO_TYPE + '/'
SATURATION_RANGE = (0.3, 2.0)
VALUE_RANGE = (0.3, 2.0)


def changeHSV(img, saturation, value):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*saturation
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return img

#file_dir = os.listdir(UNO_CARD_PATH)
#loop_range = int(150/len(file_dir))
color_array = []

for color in UNO_TYPE:
    UNO_CARD_PATH = 'Training UNO/Photos V2/' + color + '/'
    file_dir = os.listdir(UNO_CARD_PATH)
    loop_range = int(150/len(file_dir))
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
            if x >= 149:
                break
        

arr = np.array(color_array)
np.savetxt('KNN generator/points/points_3.csv', arr, fmt="%s", delimiter=",")