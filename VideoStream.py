#Frame width 1920 
#Frame height 1080
#Frame rate 30

import cv2 as cv
import numpy as np

cap = cv.VideoCapture('video.mp4')
if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()): 
    #ret är return värdet och kommer retunera true om framen har laddats korrekt
    ret, frame = cap.read()
    if ret == True:
        #Visar en frame av videon och visar denna i 25ms 
        cv.imshow('Frame',frame)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
