import cv2 as cv


def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


img = cv.imread('Photos/20230406_151337.jpg')
img_resized = rescaleFrame(img)

# Converting to grayscale

# grey = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# cv.imshow('Grey', grey)
cv.imshow('Original', img_resized)

# Blur
blur = cv.GaussianBlur(img_resized, (9, 9), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

# Edge Cascade
canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges', canny)

# Dilating the image
dilated = cv.dilate(canny, (7, 7), iterations=3)
# cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (7, 7), iterations=3)
# cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500, 500))
# cv.imshow('Resized', resized)

# Cropping
cropped = resized[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
