import numpy as np
import cv2 as cv

COLORS = ['blue', 'green', 'red', 'wild', 'yellow']

def getColor(qImg):

    mean = cv.mean(qImg)[:3]
    mean = np.asarray(mean).reshape((1,3)).astype(np.float32)
    print(mean)
    arr = np.loadtxt("Implementation/TM_KNN/allv2.csv", delimiter=",", dtype=int).astype(np.float32)
    res = np.loadtxt("Implementation/TM_KNN/resv2.csv",dtype=int).astype(np.float32)
    knn= cv.ml.KNearest_create()
    knn.train(arr, cv.ml.ROW_SAMPLE, res)
    ret, results, neighbours, dist = knn.findNearest(mean,27)

    #0: blue 
    #1: green
    #2: red
    #3: wild
    #4: yellow

    print( "result:  {}\n".format(results) )
    print( "neighbours:  {}\n".format(neighbours) )
    print( "distance:  {}\n".format(dist) )

    return COLORS[int(results)]