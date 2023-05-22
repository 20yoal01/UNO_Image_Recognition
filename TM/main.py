import cv2 as cv
import numpy as np
import card
import templateMatch

#################################
# Testning på bara en frame
#################################
img = cv.imread(r'C:\UNO Synthetic Generator\frame.jpg')

procImg = card.process(img)

cv.imshow('a', procImg)
cv.waitKey(0)

matchedCard = templateMatch.match(procImg)

print(matchedCard)
