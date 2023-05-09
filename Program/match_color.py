import numpy as np
import cv2 as cv

qImg = cv.imread('red_nine_test.jpg', cv.IMREAD_UNCHANGED)
rImg = cv.imread(r'Templates\Bin_re\green_nine.jpg', cv.IMREAD_UNCHANGED)

print(qImg.shape)
print(rImg.shape)