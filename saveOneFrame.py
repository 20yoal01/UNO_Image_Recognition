import cv2 as cv

cap = cv.VideoCapture('video.mp4')
if (cap.isOpened()== False):
    print("Error opening video stream or file")
    #ret är return värdet och kommer retunera true om framen har laddats korrekt
ret, frame = cap.read()
if ret == True:
        #Visar en frame av videon och visar denna i 25ms 
    cv.imshow('Frame',frame)
    cv.imwrite("frame.jpg", frame)
    cv.waitKey(0)
cap.release()
cv.destroyAllWindows()