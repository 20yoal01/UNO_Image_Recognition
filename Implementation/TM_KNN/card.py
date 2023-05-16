import cv2 as cv
import numpy as np

TEMPLATE_WIDTH = 225
TEMPLATE_HIGHT = 349
LOWER_THRESHOLD = 100
UPPER_THRESHOLD = 255
APERTURE_SIZE = 5

# CNN + KNN // Integrera delarna tillsammans
# TM + KNN  // Preprocessing
# TM // Inte testad
# CNN // Inte tränad


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


#TODO fixa så den tar hänsyn till att kortet är vridet 90grader
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, cnt):
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

def process(img):
    processed = img.copy()

    processed = cv.resize(processed, None, fx=0.05, fy=0.05, interpolation=cv.INTER_AREA)

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
            cards.append(four_point_transform(blank, cnt_sorted[i]))

    cv.copyTo(processed,blank,blank)

    return cards[0]