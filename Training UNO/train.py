import cv2 as cv
import numpy as np
import os

UNO_CARDS = 'OpenCV Course/Photos V2/'
IMAGES_TO_CREATE = 100
OUTPUT = 'UNO Syn/'
BACKGROUNDS = r'U:\[target_dir\validation]'


def resize(img, scale_percent=.05):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 0.8)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)
    
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

def shearX(img, shearFactor=0.1):
    M = np.float32([[1, shearFactor, 0], [0, 1, 0]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, M, dimensions)

def shearY(img, shearFactor=0.1):
    M = np.float32([[1, 0, 0], [shearFactor, 1, 0]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, M, dimensions)
    
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
    kernel = np.ones((3,3),np.uint8)
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



for filename in os.listdir('OpenCV Course/Photos V2/'):
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".jpg":
        filepath = os.path.join('OpenCV Course/Photos V2/', filename)
        img = cv.imread(filepath)
        img_resized = resize(img)
        img_resized = shearY(img_resized)
        img_resized = rotate(img_resized, 90)
        #blank = np.zeros(shape=img_resized.shape, dtype='uint8')
        #blank[:] = (255, 255, 255)
        blank = cv.imread(r'C:\Users\ejestxa\Documents\img\UNO_Image_Recognition\Training UNO\namn.jpg')
        down_width = 346
        down_height = 462
        down_points = (down_width, down_height)
        #blank = cv.resize(blank, down_points, interpolation=cv.INTER_AREA)
        mask = extract_uno(img_resized)

        pic = add_obj(blank, img_resized, mask, 200, 300)
        #h, w = mask.shape[:2]
        #x_offset = 50
        #y_offset = 0
        
        #blank[y_offset:y_offset+h, x_offset:x_offset+w] = mask
        #cv.copyTo(img_resized,mask,blank)
        
        cv.imshow("UNO", pic)
        cv.waitKey(0)
        cv.destroyAllWindows() 
