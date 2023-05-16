import cv2 as cv
import numpy as np
import card
import templateMatch
import videoStream

#################################
# Testning på bara en frame
#################################
img = cv.imread(r'WIN_20230515_14_13_23_Pro.jpg')

procImg = card.process(img)

cv.imshow('a', procImg)
cv.waitKey(0)

matchedCard = templateMatch.match(procImg)

print(matchedCard)