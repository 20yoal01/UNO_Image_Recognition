import cv2 as cv
import numpy as np
import card
import templateMatch
import videoStream

#################################
# Testning på bara en frame
#################################
img = cv.imread(r'photos\new\green\WIN_20230516_15_37_27_Pro.jpg')

procImg = card.process(img)

cv.imshow('a',procImg)
cv.waitKey(0)
matchedCard = templateMatch.match(procImg)
print(matchedCard)