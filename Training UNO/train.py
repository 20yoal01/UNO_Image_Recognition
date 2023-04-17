import cv2 as cv
import numpy as np
import os

PATH = 'UNO Syn/UNO.jpg'

def resize(img, scale_percent=.05):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def process_image(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (5,5), cv.BORDER_DEFAULT)

    t_lower = 0  # Lower Threshold
    t_upper = 255  # Upper threshold
    aperture_size = 5  # Aperture size

    canny = cv.Canny(blur, t_lower, t_upper)

    #cv.imshow('canny', canny)

    # Dilate the edges to close any gaps in the white outline
    kernel = np.ones((3,3),np.uint8)
    canny = cv.dilate(canny,kernel,iterations = 1)

    contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    #cv.imshow('Canny', canny)

    mask = np.zeros(shape=img.shape, dtype='uint8')

    cv.drawContours(mask, contours, -1, (255,255,255), thickness=cv.FILLED)

    
    #cv.imshow('A', img)

    #cv.imshow('A', mask)

    blank = np.zeros(shape=img.shape, dtype='uint8')
    blank[:] = (255, 0, 0)

    cv.copyTo(img, mask, blank)
    
    cv.rectangle(blank, (135, 145), (blank.shape[1]//5,
    blank.shape[0]//7), (0, 255, 0), thickness=3)
    cv.normalize(mask.copy(), mask, 0, 255, cv.NORM_MINMAX)

    return blank
    """ cv.imshow('Blank', blank)
    cv.imshow('Contours', mask)
    cv.imwrite(PATH, blank) """
    cv.waitKey(0)
    

    
for filename in os.listdir('OpenCV Course/Photos V2/'):
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".jpg":
        filepath = os.path.join('OpenCV Course/Photos V2/', filename)
        img = cv.imread(filepath)
        img_resized = resize(img)
        processed_image = process_image(img_resized)

        cv.imshow("UNO", processed_image)
        cv.waitKey(0)
        cv.destroyAllWindows() 
