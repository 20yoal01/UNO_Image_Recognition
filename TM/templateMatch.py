import cv2 as cv
import numpy as np
import os

COLOR_MAP_PATH = 'TM/Color/'
COLOR_TEMPLATES_PATH = ['blue_one', 'red_one', 'green_one', 'yellow_one', 'wild_wild_card', 'wild_wild_custom', 'wild_d4', 'wild_wild_shuffle', 'wild_wild_card_u', 'wild_d4_u','wild_wild_shuffle_u']
COLORS = ['blue', 'red', 'green', 'yellow', 'wild', 'wild', 'wild', 'wild', 'wild', 'wild', 'wild']
BIN_TEMPLATES_PATH = 'TM/Bin_v2/'
BKG_THRESH = 20 

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

    file_dir = os.listdir(BIN_TEMPLATES_PATH)
    for file in file_dir: 
        if file != 'yellow':
            temp = cv.imread(BIN_TEMPLATES_PATH + file, cv.IMREAD_GRAYSCALE)
            ret, thresh = cv.threshold(temp, 127, 255, cv.THRESH_BINARY)
            bin_templates.append(thresh)

    qImg_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bkg_level = cv.mean(qImg_gray)[:3]
    thresh_level = int(bkg_level[0]) + BKG_THRESH
    ret, thresh_qCard = cv.threshold(qImg_gray, thresh_level, 255, cv.THRESH_BINARY_INV)
    #cv.imshow('hej', thresh_qCard)
    bin_diff = []

    index = 0
    for tempalte_bin in bin_templates: 
        temp_diff = cv.absdiff(thresh_qCard,tempalte_bin)
        index += 1
        bin_diff.append(int(np.sum(temp_diff)/255))

    diff_num = bin_diff[np.argmax(bin_diff)]
    if (diff_num < 52_000):
        print('unknown symbol')
        return 'unknown symbol', False
    best_match_symbol = file_dir[np.argmax(bin_diff)]

    symbol = str(best_match_symbol[0:len(best_match_symbol)-4])
    
    wild_cards = ['wild_card', 'wild_custom', 'wild_shuffle', 'd4']
    isWild = False
    
    for card in wild_cards:
        if symbol == card:
            isWild = True
    
    if isWild == False:
        match = best_match_color + ' ' + str(best_match_symbol[0:len(best_match_symbol)-4])
    else:
        match = symbol.replace('_', ' ')
        
    print(match)
    return match, True