import cv2 as cv
import numpy as np
import card
import templateMatch

def resize(img, scale_percent=.5):
    y,x,c = img.shape
    return cv.resize(img, None , fx= scale_percent, fy= scale_percent, interpolation=cv.INTER_AREA)

cap = cv.VideoCapture(0)
if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()): 
    #ret är return värdet och kommer retunera true om framen har laddats korrekt
    ret, frame = cap.read()
    if ret == True:
        #frame = resize(frame)
        #Visar en frame av videon och visar denna i 25ms 
        cv.imshow('Frame', frame)
        procImg = card.process(frame)
        for pic in procImg:
            cv.imshow('a',pic)
            cv.waitKey(0)
            matchedCard = templateMatch.match(pic)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()