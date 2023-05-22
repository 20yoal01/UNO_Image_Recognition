import numpy as np
import cv2 as cv

COLORS = ['blue', 'green', 'red', 'wild', 'yellow']

def getColor(qImg):
    mean = cv.mean(qImg)[:3]
    mean = np.asarray(mean).reshape((1,3)).astype(np.float32)
    print(mean)
    arr = np.loadtxt("TM_KNN/new_all_bgr.csv", delimiter=",", dtype=int).astype(np.float32)
    res = np.loadtxt("TM_KNN/resv2.csv",dtype=int).astype(np.float32)
    knn= cv.ml.KNearest_create()
    knn.train(arr, cv.ml.ROW_SAMPLE, res)
    
    ret, results, neighbours, dist = knn.findNearest(mean, 17)

    #0: blue 
    #1: green
    #: red
    #3: wild
    #4: yellow

    print( "result:  {}\n".format(results) )
    print( "neighbours:  {}\n".format(neighbours) )
    print( "distance:  {}\n".format(dist) )

    return COLORS[int(results)]