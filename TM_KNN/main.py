import cv2 as cv
import numpy as np
import card
import templateMatch
import videoStream

#################################
#   Testning på bara en frame   #
#################################
img = cv.imread(r'WIN_20230517_14_49_10_Pro.jpg')
procImg = card.process(img)
cv.imshow('a',procImg)
cv.waitKey(0)
matchedCard = templateMatch.match(procImg)
print(matchedCard)