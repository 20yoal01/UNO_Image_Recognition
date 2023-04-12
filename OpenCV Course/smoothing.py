import cv2 as cv

def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread('OpenCV Course/Photos/20230406_151337.jpg')
img_resized = rescaleFrame(img)

#cv.imshow('UNO', img_resized)

# Averaging
average = cv.blur(img_resized, (7,7))
cv.imshow('Average Blur', average)

# Gaussian blur
gauss = cv.GaussianBlur(img_resized, (7,7), 0)
cv.imshow('Gaussian blur', gauss)

# Median blur
median = cv.medianBlur(img_resized, 7)
cv.imshow('Median blur', median)

# Bilateral
bilateral = cv.bilateralFilter(img_resized, 10, 35, 25)
cv.imshow('Bilateral', bilateral)

cv.waitKey(0)