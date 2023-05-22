import cv2 as cv
import numpy as np
import os
import knn



BIN_TEMPLATES_PATH = 'TM_KNN/Bin_v2/'
BKG_THRESH = 20

def match(img):
    bin_templates = []
    qImg = img.copy()
    best_match_color = knn.getColor(qImg)

    file_dir = os.listdir(BIN_TEMPLATES_PATH)
    for file in file_dir: 
        temp = cv.imread(BIN_TEMPLATES_PATH + file, cv.IMREAD_GRAYSCALE)
        ret, thresh = cv.threshold(temp, 127, 255, cv.THRESH_BINARY)
        bin_templates.append(thresh)

    qImg_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bkg_level = cv.mean(qImg_gray)[:3]
    thresh_level = int(bkg_level[0]) + BKG_THRESH
    ret, thresh_qCard = cv.threshold(qImg_gray, thresh_level, 255, cv.THRESH_BINARY_INV)
    cv.imshow('hej', thresh_qCard)
    bin_diff = []

    index = 0
    for tempalte_bin in bin_templates: 
        temp_diff = cv.absdiff(thresh_qCard,tempalte_bin)
        index += 1
        bin_diff.append(int(np.sum(temp_diff)/255))

    diff_num = bin_diff[np.argmax(bin_diff)]
    if (diff_num < 52_000):
        return 'unknown symbol', False
    best_match_symbol = file_dir[np.argmax(bin_diff)]

    match = best_match_color + ' ' + str(best_match_symbol[0:len(best_match_symbol)-4])
    print(match)
    return match, True