import cv2 as cv 
import numpy as np
import os

SCALE_PERCENT =.05
LOWER_THRESHOLD = 0
UPPER_THRESHOLD = 255
APERTURE_SIZE = 3
UNO_CARDS_PATH = 'photos/'
OUTPUT_GRAY = 'Templates/Gray/'
OUTPUT_COLOR = 'Templates/Color/'
CARD_TYPE = ['red', 'green', 'blue', 'yellow', 'wild']
CARD_AMOUNT = 56

#Lägger in rätt punkter på rätt plats i rektangeln
def order_points(pts):
    rect = np.zeros((4,2), dtype= "float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

#Utför en perspektiv transformation 
def four_point_transform(image, pts):
    rect = order_points(pts)
    (topL,topR,botR,botL) = rect
    wBottom = np.sqrt(((botR[0] - botL[0]) ** 2) + ((botR[1] - botL[1]) ** 2))
    wTop = np.sqrt(((topR[0] - topL[0]) ** 2) + ((topR[1] - topL[1]) ** 2))
    maxWidth = max(int(wBottom), int(wTop))
    hRight = np.sqrt(((topR[0] - botR[0]) ** 2) + ((topR[1] - botR[1]) ** 2))
    hLeft = np.sqrt(((topL[0] - botL[0]) ** 2) + ((topL[1] - botL[1]) ** 2))
    maxHeight = max(int(hRight), int(hLeft))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def extract_uno(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, LOWER_THRESHOLD, UPPER_THRESHOLD, apertureSize=APERTURE_SIZE)
    karnel = np.ones((5,5), np.uint8)
    canny = cv.dilate(canny, karnel, iterations=1)

    contours, hierarchies = cv.findContours(canny, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    return (gray, contours)


def extract_uno_thresh(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,127,255,0)
    contours, hierarchies = cv.findContours(thresh,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
    return (gray,contours)

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

    gray, contours = extract_uno_thresh(img)

    imx = img.shape[0]
    imy = img.shape[1]
    lp_area = (imx * imy) / 10

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)

        if len(approx) == 4 and cv.contourArea(cnt) > lp_area:

            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            topLeft = box[0]
            topRight = box[1]
            botRight = box[2]
            botLeft = box[3]

            pts = np.array([topLeft,topRight,botRight,botLeft])
            warped = four_point_transform(img,pts)
            warped_gray = four_point_transform(gray,pts)
            warped = cv.resize(warped, None, fx= SCALE_PERCENT, fy= SCALE_PERCENT, interpolation=cv.INTER_AREA)
            warped_gray = cv.resize(warped_gray, None, fx= SCALE_PERCENT, fy= SCALE_PERCENT, interpolation=cv.INTER_AREA)

            cv.imwrite(OUTPUT_COLOR + CARD_TYPE[index] + "_" + filename, warped)
            cv.imwrite(OUTPUT_GRAY + CARD_TYPE[index] + "_" + filename, warped_gray)

