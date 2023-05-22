import cv2 as cv
import numpy as np
import card
import templateMatch


#################################
#   Testning på bara en frame   #
#################################
img = cv.imread(r'validation bilder\WIN_20230517_14_48_48_Pro.jpg')
procImg = card.process(img)
for pic in procImg:
    cv.imshow('a',pic)
    cv.waitKey(0)
    matchedCard = templateMatch.match(pic)
print(matchedCard)