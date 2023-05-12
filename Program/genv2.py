import cv2 as cv
import numpy as np
import os

TEMPLATE_WIDTH = 225
TEMPLATE_HIGHT = 349
SCALE_PERCENT =.05
LOWER_THRESHOLD = 0
UPPER_THRESHOLD = 255
APERTURE_SIZE = 3
UNO_CARDS_PATH = 'photos/'
OUTPUT_COLOR = 'Templates/Colorv2/'
CARD_TYPE = ['red', 'green', 'blue', 'yellow', 'wild']
CARD_AMOUNT = 1

def extract_uno_thresh(img):
    processed = img.copy()
    gray = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 11)
    blur = cv.bilateralFilter(blur, 9, 130, 130)
    
    canny = cv.Canny(blur, LOWER_THRESHOLD, UPPER_THRESHOLD, apertureSize=APERTURE_SIZE)
    karnel = np.ones((5,5), np.uint8)
    canny = cv.dilate(canny, karnel,iterations=1)
    ret, threshold = cv.threshold(canny, 125, 255, cv.THRESH_BINARY)
    contours, hier = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    mask = np.zeros(shape=processed.shape, dtype=np.uint8)

    index_sort = sorted(range(len(contours)), key=lambda i : cv.contourArea(contours[i]), reverse=True)

    cnt_sorted = []
    hier_sorted = []
    cards = []

    contour_is_card = np.zeros(len(contours), dtype=int)
    
    cv.drawContours(mask, contours, -1, (255,255,255), thickness=cv.FILLED)
    
    for i in index_sort:
        cnt_sorted.append(contours[i])
        hier_sorted.append(hier[0][i])
    
    for i in range(len(contours)):
        size = cv.contourArea(cnt_sorted[i])
        peri = cv.arcLength(cnt_sorted[i],True)
        approx = cv.approxPolyDP(cnt_sorted[i],0.02*peri,True)
        if((size < 120_000) and (size > 0) and (hier_sorted[i][3] == -1) and (len(approx) == 4)):
            contour_is_card[i] = 1

    blank = np.zeros(shape=processed.shape, dtype='uint8')
    blank[:] = (0, 0, 0)

    for i in range(len(cnt_sorted)):
        if (contour_is_card[i] == 1):
            cv.drawContours(blank, [cnt_sorted[i]], -1, (255,225,225), -1)

    cv.copyTo(processed,blank,blank)

    return blank

def extract_points(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(img_gray, 500, 0.01,5)

    c = corners.sum(axis=2)
    cd = np.diff(corners,axis=2)

    tl = corners[np.argmin(c)]
    br = corners[np.argmax(c)]
    tr = corners[np.argmin(cd)]
    bl = corners[np.argmax(cd)]

    topLeft = (tl[0][0], tl[0][1])
    topRight = (tr[0][0], tr[0][1])
    botRight = (br[0][0], br[0][1])
    botLeft = (bl[0][0], bl[0][1])
    pts = np.array([topLeft,topRight,botRight,botLeft])
    return pts

def four_point_transform(image):
    pts = extract_points(image)
    maxWidth = TEMPLATE_WIDTH
    maxHeight = TEMPLATE_HIGHT
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(pts, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight), cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0))
    return warped

index = 0
folder_index = 0
current_uno_path = UNO_CARDS_PATH + CARD_TYPE[index]
total_cards = 0

while True: 
    if total_cards >= CARD_AMOUNT:
        break
    
    total_cards += 1

    filedir = os.listdir(current_uno_path)

    if folder_index >= len(filedir) and index < len(CARD_TYPE):
        index += 1
        if index >= len(CARD_TYPE):
            break
        current_uno_path = UNO_CARDS_PATH + CARD_TYPE[index]
        filedir = os.listdir(current_uno_path)
        folder_index = 0

    filename = filedir[folder_index]

    cardType = CARD_TYPE[index] + " " + filename[:]
    print(cardType)

    folder_index += 1

    file_path = os.path.join(current_uno_path, filename)
    img = cv.imread(file_path)
########################################################################
    
    img_n = extract_uno_thresh(img)
    warped = four_point_transform(img_n)
                        
    cv.imwrite(OUTPUT_COLOR + CARD_TYPE[index] + "_" + filename, warped)

