import cv2 as cv 
import numpy as np
import os

SCALE_PERCENT = 0.1
LOWER_THRESHOLD = 0
UPPER_THRESHOLD = 255
APERTURE_SIZE = 3
UNO_CARDS_PATH = 'photos/'
OUTPUT_GRAY = 'Templates/Gray'
OUTPUT_COLOR = 'Templates/Color'
CARD_TYPE = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'WILD']


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



img = cv.imread('20230406_151513.jpg')
img = cv.resize(img, None, fx= SCALE_PERCENT, fy= SCALE_PERCENT, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#blur = cv.GaussianBlur(gray, (7,7), cv.BORDER_DEFAULT)

#canny = cv.Canny(blur, LOWER_THRESHOLD, UPPER_THRESHOLD, apertureSize=APERTURE_SIZE)

ret,thresh= cv.threshold(gray,150,255,0)
cv.imshow('thresholded original',thresh)
cv.waitKey(0)

contours, hierarchies = cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

imx = img.shape[0]
imy = img.shape[1]
lp_area = (imx * imy) / 10

#for cnt in contours:
#    approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
#
#    if len(approx) == 4 and cv.contourArea(cnt) > lp_area:
#        tmp_img = img.copy()
#        cv.drawContours(tmp_img, [cnt], 0, (0,255,255),6)
#        #cv.imshow('Contour Borders', tmp_img)
#        #cv.waitKey(0)
#
#        tmp_img = img.copy()
#        mask = np.zeros((img.shape[:2]),np.uint8)
#        hull = cv.convexHull(cnt)
#        cv.drawContours(mask, [hull],0,(255,255,255),-1)
#        #cv.imshow('hull',mask)
#        #cv.waitKey(0)
#
#        tmp_img = img.copy()
#        rect = cv.minAreaRect(cnt)
#        box = cv.boxPoints(rect)
#        box = np.int0(box)
#        print(box)
#        #cv.drawContours(tmp_img, [box],0,(0,0,255),2)
#        #cv.waitKey(0)
#
#        tmp_img = img.copy()
#        topLeft = box[0]
#        topRight = box[1]
#        botRight = box[2]
#        botLeft = box[3]
#        cv.drawContours(tmp_img, [cnt], -1, (0, 255, 255), 2)
#        cv.circle(tmp_img, topLeft, 8, (0, 0, 255), -1)
#        cv.circle(tmp_img, topRight, 8, (0, 255, 0), -1)
#        cv.circle(tmp_img, botRight, 8, (255, 0, 0), -1)
#        cv.circle(tmp_img, botLeft, 8, (255, 255, 0), -1)
#        #cv.imshow('img contour drawn', tmp_img)
#        #cv.waitKey(0)
#
#        tmp_img = img.copy()
#        pts = np.array([topLeft,topRight,botRight,botLeft])
#        warped = four_point_transform(tmp_img,pts)
#        #cv.imshow("Warped",warped)
#        #cv.waitKey(0)
#        cv.imwrite('test.jpg',warped)

#cv.waitKey(0)