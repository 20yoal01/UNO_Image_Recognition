import cv2 as cv
import numpy as np
import os
import random

UNO_CARDS_PATH = 'OpenCV Course/Photos V2/'
OUTPUT = 'UNO Syn/'
BACKGROUND_PATH = 'OpenCV Course/background'
CARD_TYPE = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'WILD']


IMAGES_PER_CARD = 100
TOTAL_CARDS_TO_GENERATE = 1


TARGET_HEIGHT, TARGET_WIDTH = (720, 1280)
ROTATE_RANGE = [-180, 180]
TRANSLATE_RANGE = [(TARGET_WIDTH//2)*0.06, (TARGET_HEIGHT//2)*0.06] # 10% of screen
SHEAR_RANGE = (-0.35, 0.35)
PROJECTION_RANGE = (-0.35, 0.35)
SATURATION_RANGE = 0
VALUE_RANGE = 0

def resize(img, scale_percent=.05):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def resize_no_aspect(img, width, height):
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

# def rotate(img, angle, rotPoint=None):
#     (height, width) = img.shape[:2]

#     if rotPoint is None:
#         rotPoint = (width//2, height//2)

#     rotMat = cv.getRotationMatrix2D(rotPoint, angle, 0.8)
#     dimensions = (width, height)

#     return cv.warpAffine(img, rotMat, dimensions)
    
# def translate(img, x, y):
#     transMat = np.float32([[1, 0, x], [0, 1, y]])
#     dimensions = (img.shape[1], img.shape[0])
#     return cv.warpAffine(img, transMat, dimensions)
    

def shearX(img, shearFactor=0.1):
    M = np.float32([[1, shearFactor, 0], [0, 1, 0]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, M, dimensions)
def shearY(img, shearFactor=0.1):
    M = np.float32([[1, 0, 0], [shearFactor, 1, 0]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, M, dimensions)

def projectiveTransform(img, angle, translation=(0, 0), shear=(0,0), projective=(0,0), rotPoint=None):
    (height, width) = img.shape[:2]
    dimensions = (width, height)
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    
    # Roterar bilden
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    rotMat = np.vstack([rotMat, [0, 0, 1]])

    # Shear
    shearMat = np.float32([[1, shear[0], 0], [shear[1], 1, 0], [0, 0, 1]])

    # Translation
    transMat = np.float32([[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])
    
    # Beräknar våran nya perspektiv matris 
    src_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_pts = np.float32([[0, 0], [width, 0], [width + projective[0] * height, height], [-projective[1] * width, height]])
    perspMat = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Kombinerar allting
    projMat = np.dot(transMat,  np.dot(shearMat, np.dot(perspMat, rotMat)))
    
    return cv.warpPerspective(img, projMat, dimensions)

#def changeHSV():
    
    
    
def extract_uno(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (5,5), cv.BORDER_DEFAULT)

    t_lower = 0  # Lower Threshold
    t_upper = 255  # Upper threshold
    aperture_size = 3  # Aperture size

    canny = cv.Canny(blur, t_lower, t_upper, apertureSize=aperture_size)

    #cv.imshow('canny', canny)

    # Dilate the edges to close any gaps in the white outline
    kernel = np.ones((5,5),np.uint8)
    canny = cv.dilate(canny,kernel,iterations = 1)

    contours, hierarchies = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 
    #cv.imshow('Canny', canny)

    mask = np.zeros(shape=img.shape, dtype='uint8')

    cv.drawContours(mask, contours, -1, (255,255,255), thickness=cv.FILLED)
    x, y, w, h = cv.boundingRect(contours[0])

    blank = np.zeros(shape=img.shape, dtype='uint8')
    #blank[:] = (255, 255, 255)
    #cv.imshow('sh', mask)
    #mask_cropped = mask[y:y+h, x:x+w]
    #masked_img = img[y:y+h, x:x+w]
    cv.copyTo(img, mask, blank)
    #cv.imshow('m', blank)  
    #cv.rectangle(blank, (135, 145), (blank.shape[1]//5,
    #blank.shape[0]//7), (0, 255, 0), thickness=3)
    #cv.normalize(mask.copy(), mask, 0, 255, cv.NORM_MINMAX)
    return mask

def add_obj(background, img, mask, x, y):
    '''
    Argument: 
    background - bakgrunden som ska användas
    img - bilden på UNO-kortet "orginalet" 
    mask - masken som tog fram av tidigare metod 
    x,y - koordinaterna för mitten på bildobjektet. Dessa måste vara mindre än bakgrundens dimentioner
    '''

    bg = background.copy()

    h_bg, w_bg = bg.shape[0], bg.shape[1]
    h, w = img.shape[0], img.shape[1]

    x = x - int(w/2)
    y = y - int(h/2)

    mask_boolean = mask[:,:,0] != 0
    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)

    if x >= 0 and y >= 0:
    
        h_part = h - max(0, y+h-h_bg) # h_part - part of the image which overlaps background along y-axis
        w_part = w - max(0, x+w-w_bg) # w_part - part of the image which overlaps background along x-axis

        bg[y:y+h_part, x:x+w_part, :] = bg[y:y+h_part, x:x+w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part, :] + (img * mask_rgb_boolean)[0:h_part, 0:w_part, :]
    
    
    return bg


index = 0
folder_index = 0
CURRENT_UNO_PATH = UNO_CARDS_PATH + CARD_TYPE[index]
total_cards = 0
while True:
    if total_cards >= TOTAL_CARDS_TO_GENERATE:
        break
    
    total_cards += 1
    
    filedir = os.listdir(CURRENT_UNO_PATH)
    
    if folder_index >= len(filedir) and index < len(CARD_TYPE):
        index += 1
        if index >= len(CARD_TYPE):
            break
        CURRENT_UNO_PATH = UNO_CARDS_PATH + CARD_TYPE[index]
        filedir = os.listdir(CURRENT_UNO_PATH)
        folder_index = 0
    
    filename = filedir[folder_index]    
    
    cardType = CARD_TYPE[index] + " " + filename[:]
    print(cardType)
    
    folder_index += 1
    
    img_to_create = 0
    
    for background in os.listdir(BACKGROUND_PATH):
        if img_to_create >= IMAGES_PER_CARD:
            break
        
        img_to_create += 1
        
        ext = os.path.splitext(filename)[-1].lower()
        if ext == ".jpg":
            background_img_path = os.path.join(BACKGROUND_PATH, background)
            
            rotation     = round(random.uniform(ROTATE_RANGE[0], ROTATE_RANGE[1]), 2)
            translate_x  = round(random.uniform(-TRANSLATE_RANGE[0], TRANSLATE_RANGE[0]), 2)
            translate_y  = round(random.uniform(-TRANSLATE_RANGE[1], TRANSLATE_RANGE[1]), 2)
            shear_x      = round(random.uniform(SHEAR_RANGE[0], SHEAR_RANGE[1]), 2)
            shear_y      = round(random.uniform(SHEAR_RANGE[0], SHEAR_RANGE[1]), 2)
            projection_x = round(random.uniform(PROJECTION_RANGE[0], PROJECTION_RANGE[1]), 2)
            projection_y = round(random.uniform(PROJECTION_RANGE[0], PROJECTION_RANGE[1]), 2)
            saturation   = round(random.uniform(SATURATION_RANGE, SATURATION_RANGE), 2)
            value        = round(random.uniform(VALUE_RANGE, VALUE_RANGE), 2)

            print(rotation, translate_x, translate_y, shear_x, shear_y, projection_x, projection_y, saturation, value)

            filepath = os.path.join(CURRENT_UNO_PATH, filename)
            img = cv.imread(filepath)
            img_resized = resize(img, scale_percent=0.025)

            blank = np.zeros(shape=(TARGET_HEIGHT, TARGET_WIDTH, 3), dtype='uint8')
            blank[:] = (0, 0, 0)
            
            background_img = cv.imread(background_img_path)
            b_dimensions = background_img.shape[:2]

            background_img = resize_no_aspect(background_img, TARGET_WIDTH, TARGET_HEIGHT)
            mask = extract_uno(img_resized)
            #cv.imshow('m', mask)
            pic = add_obj(blank, img_resized, mask, TARGET_WIDTH//2, TARGET_HEIGHT//2)
            pic = projectiveTransform(pic, rotation, (translate_x, translate_y), (shear_x, shear_y), (projection_x, projection_y))

            pic_gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
            ret, pic_mask = cv.threshold(pic_gray, 10, 255, cv.THRESH_BINARY)
            background_mask = cv.bitwise_not(pic_mask)
            card_masked = cv.bitwise_and(pic, pic, mask=pic_mask)
            background_masked = cv.bitwise_and(background_img, background_img, mask=background_mask)
            pic = cv.add(card_masked, background_masked)

            cv.imwrite('test/' + background + '_' + cardType, pic)
            #cv.imshow("UNO", pic)
            cv.waitKey(0)
            cv.destroyAllWindows()
