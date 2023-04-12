import cv2 as cv

def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('Face/face1.jpg')
img_resized = rescaleFrame(img, 0.20) 

#cv.imshow('Person', img_resized)

gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number of faces found: {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img_resized, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img_resized)
cv.waitKey(0)