import cv2 as cv 
import numpy as np
import os

UNO_CARDS_PATH = 'Templates/Gray/'
OUTPUT_GRAY = 'Templates/Gray_re/'

file_dir = os.listdir(UNO_CARDS_PATH)

for file in file_dir:
    file_path = os.path.join(UNO_CARDS_PATH,file)
    img = cv.imread(file_path)

    img = cv.resize(img, (225,349), 0, 0)

    cv.imwrite(OUTPUT_GRAY + file,img)