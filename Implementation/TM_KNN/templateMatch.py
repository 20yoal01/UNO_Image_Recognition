import cv2 as cv
import numpy as np
import os
import knn

BIN_TEMPLATES_PATH = 'Templates/Bin_v2/'

def getThreshold(frame):
    mono = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.blur(mono, (5, 5))
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 255, 2)
    return th

def threshold_uno_card(image):
    # Convert image to HSV color space
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define color ranges for UNO card symbols (red, green, blue, yellow)
    color_ranges = [
        ((0, 70, 50), (10, 255, 255)),        # Red (lower range)
        ((160, 70, 50), (180, 255, 255)),     # Red (upper range)
        ((40, 70, 50), (80, 255, 255)),       # Green
        ((100, 70, 50), (130, 255, 255)),     # Blue
        ((20, 70, 50), (35, 255, 255))        # Yellow
    ]
    # Apply color thresholding for each color range
    thresholded_images = []
    for (lower, upper) in color_ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv.inRange(hsv_image, lower, upper)
        thresholded_images.append(mask)

    # Combine the thresholded images to get the final result
    result = sum(thresholded_images)

    return result

def match(img):
    bin_templates = []

    qImg = img.copy()

    best_match_color = knn.getColor(qImg)

    bin_path = BIN_TEMPLATES_PATH
    
    #if best_match_color == 'yellow':
    #    file_dir = os.listdir(BIN_TEMPLATES_PATH)
    #    
    #else:
    #    bin_path += 'yellow/'
    #    file_dir = os.listdir(BIN_TEMPLATES_PATH + 'yellow')

    file_dir = os.listdir(BIN_TEMPLATES_PATH)
    
    for file in file_dir: 
        if file != 'yellow':
            temp = cv.imread(bin_path + file, cv.IMREAD_GRAYSCALE)
            ret, thresh = cv.threshold(temp, 127, 255, cv.THRESH_BINARY)
            bin_templates.append(thresh)

    #qImg_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #ret, thresh_qCard = cv.threshold(qImg_gray, 127, 255, cv.THRESH_BINARY_INV)
    cv.imshow('red', img)
    thresh_qCard = threshold_uno_card(img)
    cv.imshow('abc', thresh_qCard)
    cv.waitKey(0)
    
    bin_diff = []

    index = 0

    for tempalte_bin in bin_templates: 
        temp_diff = cv.absdiff(thresh_qCard,tempalte_bin)
        #cv.imshow(str(index),temp_diff)
        index += 1
        bin_diff.append(int(np.sum(temp_diff)/255))

    print('diff')
    print(np.argmax(bin_diff))
    best_match_symbol = file_dir[np.argmax(bin_diff)]

    match = best_match_color + ' ' + str(best_match_symbol[0:len(best_match_symbol)-4])
    print(bin_diff)
    print(file_dir)
    print(match)
    cv.waitKey(0)
    return match