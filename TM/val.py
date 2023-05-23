import os
import cv2 as cv
import numpy as np

file_dir = os.listdir('validation bilder')
result = []
for file in file_dir:
    print("Färg: ", file[:file.find('_')])
    print("Kort: ", file[file.find('_')+1:len(file)-6])
    print(' ')
    

file_dir = os.listdir('TM_KNN/Bin_v2')
print(file_dir)
for file in file_dir:
    if file.find('_upsidedown') != -1:
        print("Kort med _: ", file[:file.find('_upsidedown')])
    else: 
        print("Kort utan: ",file[:len(file)-4])
    print(" ")


#Finns det ett kort? Från cards
#Vilken Färg, från template 
#Vilken symbol, template
#CSV: facit färg, facit symbol, finns kort (1/0), färg, symbol

img = cv.imread(r'validation bilder\WIN_20230517_14_48_48_Pro.jpg')
procImg = card.process(img)
for pic in procImg:
    cv.imshow('a',pic)
    cv.waitKey(0)
    matchedCard = templateMatch.match(pic)
print(matchedCard)