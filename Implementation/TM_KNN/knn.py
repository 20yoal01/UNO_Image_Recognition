import numpy as np
import cv2 as cv

COLORS = ['blue', 'green', 'red', 'wild', 'yellow']

def hisEqulColor(img):
    # convert from RGB color-space to YCrCb
    ycrcb_img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv.equalizeHist(ycrcb_img[:, :, 0])

    # convert back to RGB color-space from YCrCb
    equalized_img = cv.cvtColor(ycrcb_img, cv.COLOR_YCrCb2BGR)
    return equalized_img


def getColor(qImg):
    cv.imshow('img', qImg)
    qImg = hisEqulColor(qImg)
    cv.imshow('4', qImg)
    cv.waitKey(0)
    mean = cv.mean(qImg)[:3]
    mean = np.asarray(mean).reshape((1,3)).astype(np.float32)
    print(mean)
    arr = np.loadtxt("Implementation/TM_KNN/all_in.csv", delimiter=",", dtype=int).astype(np.float32)
    res = np.loadtxt("Implementation/TM_KNN/resv2.csv",dtype=int).astype(np.float32)
    knn= cv.ml.KNearest_create()
    knn.train(arr, cv.ml.ROW_SAMPLE, res)
    
    data = np.reshape(qImg, (-1,3))
    print(data.shape)
    data = np.float32(data)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv.kmeans(data,1,None,criteria,10,flags)
    
    print('mean: ', mean)
    print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))
    
    ret, results, neighbours, dist = knn.findNearest(mean,7)

    #0: blue 
    #1: green
    #: red
    #3: wild
    #4: yellow

    print( "result:  {}\n".format(results) )
    print( "neighbours:  {}\n".format(neighbours) )
    print( "distance:  {}\n".format(dist) )

    return COLORS[int(results)]