import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(r'C:\Users\ejestxa\Documents\img\UNO_Image_Recognition\20230406_151513.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, None, fx= 0.05, fy= 0.05, interpolation=cv.INTER_AREA)
dummy, img = cv.threshold(img, 150,255,0)
template = cv.imread(r'Templates\Bin\red_nine.jpg',cv.IMREAD_GRAYSCALE)

print(template.shape)

(w, h) = template.shape

methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']


res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)

threshold = 0.85
locations = np.where(res >= threshold)
locations = list(zip(*locations[::-1]))

print(len(locations))

if locations:
    print('found needle')

    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + h, top_left[1] + w)
        cv.rectangle(img, top_left,bottom_right, (255,255,255), 2)
        cv.imshow('img',img)
        cv.waitKey(0)


#for meth in methods:
#    img = img.copy()
#
#    method = eval(meth)
#
#    res = cv.matchTemplate(img, template, method)
#
#
#    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
#    print(min_val)
#    print(max_val)
#    print(max_loc)
#    print(min_loc)
#
#    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
#        top_left = min_loc
#    else:
#        top_left = max_loc
#
#    bottom_right = (top_left[0] + h, top_left[1] + w)
#
#    cv.rectangle(img,top_left, bottom_right, (255,255,255), 2)
#    cv.rectangle(img,(38,0),(59,12),(0,255,0),3)
#    cv.imshow('img',img)
#    cv.waitKey(0)
#plt.subplot(121),plt.imshow(res,cmap = 'gray')
#plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img,cmap = 'gray')
#plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#plt.suptitle(meth)
#plt.show()
