import cv2 as cv
import numpy as np
import TM.card as card
import TM.templateMatch as templateMatch
import time
import os

class videoStream:
    def __init__(self):
        print('Initializing TM KNN..')

    def resize(self, img, scale_percent=.5):
        y,x,c = img.shape
        return cv.resize(img, None , fx= scale_percent, fy= scale_percent, interpolation=cv.INTER_AREA)

    def __call__(self):
        cap = cv.VideoCapture(0)
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        frameCount = 0
        while(cap.isOpened()): 
            #ret är return värdet och kommer retunera true om framen har laddats korrekt
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if ret == True:
                #frame = resize(frame)
                #Visar en frame av videon och visar denna i 25ms 
                frameCount += 1
                if frameCount > 0:
                    #frameCount = 0
                    #cv.imshow('Frame', frame)
                    procImg, pts  = card.process(frame)
                    for i in range(len(procImg)):
                        #cv.imshow('a',procImg[i])
                        matchedCard, match_found = templateMatch.match(procImg[i])
                        if pts and match_found:
                            rect = cv.minAreaRect(pts[i])
                            box = cv.boxPoints(rect)
                            box = np.int0(box)
                            frame = cv.drawContours(frame,[box],0,(0,255,255),2)
                            frame = cv.putText(frame,matchedCard,[box[0][0],box[0][1]-10], cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                    end_time = time.perf_counter()
                    fps = 1 / np.round(end_time - start_time, 3)
                    cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 1)
                    cv.imshow('Frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv.destroyAllWindows()
        
    def time_measure(self, path):
        cap = cv.VideoCapture(path)
        video_lengh = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        time_list = []
        
        frameCount = 0
        
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        
        print(video_lengh)
        
        for i in range(video_lengh):
            #den kommer vara beroende på hur många kort som faktiskt hittas samt
            #om den inte hittar något kort alls, hur ska vi veta vad?
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if ret == True:
                print(i)
                #frame = resize(frame)
                #Visar en frame av videon och visar denna i 25ms 
                frameCount += 1
                if frameCount > 30:
                    frameCount = 0
                    #cv.imshow('Frame', frame)
                    procImg, pts  = card.process(frame)
                    for i in range(len(procImg)):
                        #cv.imshow('a',procImg[i])
                        matchedCard, match_found = templateMatch.match(procImg[i])
                        if pts and match_found:
                            rect = cv.minAreaRect(pts[i])
                            box = cv.boxPoints(rect)
                            box = np.int0(box)
                            frame = cv.drawContours(frame,[box],0,(0,255,255),2)
                            frame = cv.putText(frame,matchedCard,[box[0][0],box[0][1]-10], cv.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                    end_time = time.perf_counter()
                    time_ms = round(((end_time - start_time) / 1000), 5)
                    time_list.append(time_ms)
                    fps = 1 / np.round(end_time - start_time, 10)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        average_time = sum(time_list) / len(time_list)
        print(average_time)

    def precision_measure(self):
        return None