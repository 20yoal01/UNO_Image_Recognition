import cv2 as cv
import numpy as np

COLOR_MAP_PATH = 'Templates/Color_re/'
COLOR_TEMPLATES_PATH = ['blue_one', 'red_one', 'green_one', 'yellow_one', 'wild_wild_card']
COLORS = ['blue', 'red', 'green', 'yellow', 'wild']

def match(img):
    qImg = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype('float32')
    
    color_templates = []

    for index in range(len(COLOR_TEMPLATES_PATH)):
        color_templates.append(cv.imread(COLOR_MAP_PATH + COLOR_TEMPLATES_PATH[index] +'.jpg', cv.IMREAD_UNCHANGED))

    colorDiff = []

    for template in color_templates: 
        template = cv.cvtColor(template, cv.COLOR_BGR2HSV).astype('float32')
        temp_diff = cv.absdiff(qImg,template)
        colorDiff.append(int(np.sum(temp_diff[:,:,0])/360))

    best_match = COLORS[np.argmin(colorDiff)]
    
    


    return 