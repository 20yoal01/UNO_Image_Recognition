import numpy as np
import cv2 as cv

qImg = cv.imread('red_nine_test.jpg', cv.IMREAD_GRAYSCALE)
rImg = cv.imread(r'Templates\Bin\blue_nine.jpg', cv.IMREAD_GRAYSCALE)
wImg = cv.imread(r'Templates\Bin\red_s.jpg', cv.IMREAD_GRAYSCALE)
qImgw = cv.resize(qImg, (225,349), 0, 0)
qImgr = cv.resize(qImg, (225,349), 0, 0)

ret, thresh_qr = cv.threshold(qImgr, 127, 255, cv.THRESH_BINARY_INV)
cv.imshow('thresh qCard', thresh_qr)

ret, thresh_r = cv.threshold(rImg, 127, 255, 0)
cv.imshow('Right thresh',thresh_r)

diff_img = cv.absdiff(thresh_r,thresh_qr)
cv.imshow('diff right', diff_img)

ret, thresh_qw = cv.threshold(qImgw, 127, 255, cv.THRESH_BINARY_INV)

ret, thresh_w = cv.threshold(wImg, 127, 255, 0)
cv.imshow('Wrong thresh',thresh_w)

diff_img_w = cv.absdiff(thresh_w,thresh_qw)
cv.imshow('diff wrong', diff_img_w)

print(diff_img)

rank_diff = int(np.sum(diff_img)/255)
print(rank_diff)

rank_diff_w = int(np.sum(diff_img_w)/255)
print(rank_diff_w)

cv.waitKey(0)
cv.destroyAllWindows()