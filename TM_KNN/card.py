import cv2 as cv
import numpy as np

IMG_WIDTH = 960
IMG_HIGHT = 540
TEMPLATE_WIDTH = 225
TEMPLATE_HIGHT = 349
LOWER_THRESHOLD = 50
UPPER_THRESHOLD = 210
APERTURE_SIZE = 5
BKG_THRESH = 20

# CNN + KNN // Integrera delarna tillsammans
# TM + KNN  // Preprocessing
# TM // Inte testad
# CNN // Inte tränad


def extract_points(img, mask):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(img_gray, 500, 0.01, 5)

    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(int(x),int(y)),3,(0,0,255),-1)

    cv.imshow('img', img)
    cv.waitKey(0)

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

def order_points(pts):
    c = pts.sum(axis=1)
    cd = np.diff(pts,axis=1)

    tl = pts[np.argmin(c)]
    br = pts[np.argmax(c)]
    tr = pts[np.argmin(cd)]
    bl = pts[np.argmax(cd)]

    topLeft = (tl[0], tl[1])
    topRight = (tr[0], tr[1])
    botRight = (br[0], br[1])
    botLeft = (bl[0], bl[1])
    return topLeft,topRight,botRight,botLeft

def extract_four_points(img, cnt):
    img_controll = img.copy()
    same_x_bl = []
    same_x_tr = []
    bl = tuple(cnt[cnt[:, :, 0].argmin()][0])
    for point in cnt:
        if (point[0][0] == bl[0]):
            same_x_bl.append(point[0])
    same_x_bl.sort(key=lambda point: point[1], reverse=True)
    bl = same_x_bl[0]
    tr = tuple(cnt[cnt[:, :, 0].argmax()][0])
    for point in cnt:
        if (point[0][0] == tr[0]):
            same_x_tr.append(point[0])
    same_x_tr.sort(key=lambda point: point[1], reverse=False)
    tr = same_x_tr[0]
    same_y_br = []
    same_y_tl = []
    tl = tuple(cnt[cnt[:, :, 1].argmin()][0])
    for point in cnt:
        if (point[0][1] == tl[1]):
            same_y_tl.append(point[0])
    same_y_tl.sort(key=lambda point: point[0], reverse=False)
    tl = same_y_tl[0]
    br = tuple(cnt[cnt[:, :, 1].argmax()][0])
    for point in cnt:
        if (point[0][1] == br[1]):
            same_y_br.append(point[0])
    same_y_br.sort(key=lambda point: point[0], reverse=True)
    br = same_y_br[0]
    arr = np.array((bl,tr,tl,br))
    tl,tr,br,bl = order_points(arr)
    width = np.sqrt((tl[0]-tr[0]) **2 + (tl[1]-tr[1]) **2)
    hight = np.sqrt((tr[0]-br[0]) **2 + (tr[1]-br[1]) **2)
    if(width < hight):
        topLeft = (tl[0], tl[1])
        topRight = (tr[0], tr[1])
        botRight = (br[0], br[1])
        botLeft = (bl[0], bl[1])
    else:
        topRight = (tl[0], tl[1])
        botRight = (tr[0], tr[1])
        botLeft = (br[0], br[1])
        topLeft = (bl[0], bl[1])
    cv.circle(img_controll, botLeft, 8, (0, 0, 255), -1)
    cv.circle(img_controll, topRight, 8, (0, 255, 0), -1)
    cv.circle(img_controll, topLeft, 8, (255, 0, 0), -1)
    cv.circle(img_controll, botRight, 8, (255, 255, 0), -1)
    cv.imshow('test', img_controll)
    pts = np.array([topLeft,topRight,botRight,botLeft]).astype(np.float32)
    return pts

def four_point_transform(image, cnt):
    pts = extract_four_points(image,cnt)
    maxWidth = TEMPLATE_WIDTH
    maxHeight = TEMPLATE_HIGHT
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv.getPerspectiveTransform(pts, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process(img):
    processed = img.copy()
    img_h, img_w = np.shape(img)[:2]
    scale_x = IMG_WIDTH/img_w
    scale_y = IMG_HIGHT/img_h

    processed = cv.resize(processed, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_AREA)
    print(processed.shape)
    gray = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 11)
    blur = cv.bilateralFilter(blur, 11, 150, 150)
    mean_value = np.mean(blur)
    lower_threshold = int(max(0, mean_value - 2 * np.std(blur)))
    upper_threshold = int(min(255, mean_value + 2 * np.std(blur)))
    canny = cv.Canny(blur, lower_threshold, upper_threshold, apertureSize=APERTURE_SIZE)
    karnel = np.ones((3,3), np.uint8)
    canny = cv.dilate(canny, karnel,iterations=1)
    cv.imshow('canny', canny)
    bkg_level = cv.mean(gray)[:3]
    thresh_level = int(bkg_level[0]) + BKG_THRESH
    ret, threshold = cv.threshold(canny, thresh_level, 255, cv.THRESH_BINARY)
    contours, hier = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    mask = np.zeros(shape=processed.shape[:2], dtype=np.uint8)

    index_sort = sorted(range(len(contours)), key=lambda i : cv.contourArea(contours[i]), reverse=True)

    cnt_sorted = []
    hier_sorted = []
    cards = []

    contour_is_card = np.zeros(len(contours), dtype=int)
    print('array: ' + str(len(contour_is_card)))

    
    for i in index_sort:
        cnt_sorted.append(contours[i])
        hier_sorted.append(hier[0][i])
    
    for i in range(len(contours)):
        size = cv.contourArea(cnt_sorted[i])
        peri = cv.arcLength(cnt_sorted[i],True)
        approx = cv.approxPolyDP(cnt_sorted[i],0.02*peri,True)
        if((size < 120_000_000) and (size > 1_200) and (hier_sorted[i][3] == -1)):
            contour_is_card[i] = 1
    
    print(contour_is_card)

    blank = np.zeros(shape=processed.shape, dtype='uint8')
    blank[:] = (0, 0, 0)

    for i in range(len(cnt_sorted)):
        if (contour_is_card[i] == 1):
            cv.drawContours(mask, [cnt_sorted[i]], -1, (255,225,225), -1)
            mask = cv.erode(mask, None, iterations=2)
            cv.imshow('mask', mask)
            cards.append(four_point_transform(processed, cnt_sorted[i]))

    return cards