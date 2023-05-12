import cv2 as cv 
import numpy as np
import os

UNO_CARDS_PATH = 'Templates/yellowv2/'
OUTPUT_GRAY = 'Templates/Bin_y_v2/'

file_dir = os.listdir(UNO_CARDS_PATH)

for file in file_dir:
    file_path = os.path.join(UNO_CARDS_PATH,file)
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

    img = cv.resize(img, (225,349), interpolation=cv.INTER_AREA)

    ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    cv.imwrite(OUTPUT_GRAY + file,img)