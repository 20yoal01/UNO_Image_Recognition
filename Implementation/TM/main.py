import cv2 as cv
import numpy as np
import card
import templateMatch
import videoStream

#################################
# Testning på bara en frame
#################################
img = cv.imread(r'C:\Users\ejestxa\Documents\img\UNO_Image_Recognition\frame.jpg')

procImg = card.process(img)

matchedCard = templateMatch.match(procImg)

print(matchedCard)