import YOLOv5_UNO.CNN_KNN.detect_CNN_KNN as detect_CNN_KNN
import YOLOv5_UNO.CNN.detect_CNN as detect_CNN
import TM.videoStream as videoStream_TM
import TM_KNN.videoStream as videoStream_TM_KNN
import threading

print('UNO Card Detector press one of these numbers to start!')
print('1: Template Matching with Color')
print('2: Template Matching with KNN')
print('3: CNN with Color')
print('4: CNN with KNN')

#
value = None

while True:
    value = input()
    try:
       value = int(value)
    except ValueError:
       print ('Invalid range, please enter: 1-4')
       continue
    if 1 <= value <= 4:
       break
    else:
       print ('Invalid range, please enter: 1-4')
       
match(value):
    case 1:
        detection = videoStream_TM.videoStream()
        detection()
    case 2:
        detection = videoStream_TM_KNN.videoStream()
        detection()
    case 3:
        detection = detect_CNN.ObjectDetection()
        detection()
    case 4:
        detection = detect_CNN_KNN.ObjectDetection()
        detection()
