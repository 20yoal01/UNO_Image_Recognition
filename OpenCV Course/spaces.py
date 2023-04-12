import cv2 as cv


def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('Photos/20230406_151337.jpg')
img_resized = rescaleFrame(img)

cv.imshow('UNO', img_resized)

# BGR to Grayscale

gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
#cv.imshow('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img_resized, cv.COLOR_BGR2HSV)
#cv.imshow('HSV', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img_resized, cv.COLOR_BGR2LAB)
#cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
#cv.imshow('RGB', rgb)

# HSV to BGR 
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
cv.imshow('HSV --> BGR', hsv_bgr)

cv.waitKey(0)
