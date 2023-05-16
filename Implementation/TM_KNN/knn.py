import numpy as np
import cv2 as cv

COLORS = ['blue', 'green', 'red', 'wild', 'yellow']

def getColor(qImg):

    mean = cv.mean(qImg)[:3]
    mean = np.asarray(mean).reshape((1,3)).astype(np.float32)
    print(mean)
    arr = np.loadtxt("Implementation/TM_KNN/all_bgr.csv", delimiter=",", dtype=int).astype(np.float32)
    knn_train = arr[:, :3]
    knn_val = arr[:,3]
    knn= cv.ml.KNearest_create()
    knn.train(knn_train, cv.ml.ROW_SAMPLE, knn_val)
    ret, results, neighbours, dist = knn.findNearest(mean,27)

    #0: blue 
    #1: green
    #2: red
    #3: wild
    #4: yellow

    print( "result:  {}\n".format(results) )
    print( "neighbours:  {}\n".format(neighbours) )
    print( "distance:  {}\n".format(dist) )

    return COLORS[results]