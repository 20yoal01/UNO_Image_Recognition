import numpy as np
import cv2 as cv

TEMPLATE_WIDTH = 225
TEMPLATE_HIGHT = 349

qImg = cv.imread(r'Templates\Color_re\yellow_one.jpg', cv.IMREAD_UNCHANGED)
rImg = cv.imread(r'Templates\Color_re\wild_wild_card.jpg', cv.IMREAD_UNCHANGED)

qImg = cv.cvtColor(qImg, cv.COLOR_BGR2HSV).astype("float32")
rImg = cv.cvtColor(rImg, cv.COLOR_BGR2HSV).astype("float32")

diff = cv.absdiff(qImg,rImg)
rank_diff = int(np.sum(diff[:,:,0])/360)
print(rank_diff)

cv.imshow('Diff',diff)
cv.waitKey(0)

print(qImg.shape)
print(rImg.shape)