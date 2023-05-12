import cv2 as cv
import numpy as np
import os

COLOR_MAP_PATH = 'Templates/Color/'
COLOR_TEMPLATES_PATH = ['blue_one', 'red_one', 'green_one', 'yellow_one', 'wild_wild_card', 'wild_wild_custom', 'wild_d4', 'wild_wild_shuffle']
COLORS = ['blue', 'red', 'green', 'yellow', 'wild', 'wild', 'wild', 'wild']
BIN_TEMPLATES_PATH = 'Templates/Bin_v2/'

def match(img):
    color_templates = []
    color_diff = []
    bin_templates = []

    qImg = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype('float32')
    for index in range(len(COLOR_TEMPLATES_PATH)):
        color_templates.append(cv.imread(COLOR_MAP_PATH + COLOR_TEMPLATES_PATH[index] +'.jpg', cv.IMREAD_UNCHANGED))

    for template in color_templates: 
        template = cv.cvtColor(template, cv.COLOR_BGR2HSV).astype('float32')
        temp_diff = cv.absdiff(qImg,template)
        color_diff.append(int(np.sum(temp_diff[:,:,0])/360))

    best_match_color = COLORS[np.argmin(color_diff)]

    bin_path = BIN_TEMPLATES_PATH
    
    if best_match_color != 'yellow':
        file_dir = os.listdir(BIN_TEMPLATES_PATH)
        
    else:
        bin_path += 'yellow/'
        file_dir = os.listdir(BIN_TEMPLATES_PATH + 'yellow')

    for file in file_dir: 
        if file != 'yellow':
            temp = cv.imread(bin_path + file, cv.IMREAD_GRAYSCALE)
            ret, thresh = cv.threshold(temp, 127, 255, cv.THRESH_BINARY)
            bin_templates.append(thresh)

    qImg_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh_qCard = cv.threshold(qImg_gray, 127, 255, cv.THRESH_BINARY_INV)

    bin_diff = []

    index = 0

    for tempalte_bin in bin_templates: 
        temp_diff = cv.absdiff(thresh_qCard,tempalte_bin)
        cv.imshow(str(index),temp_diff)
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