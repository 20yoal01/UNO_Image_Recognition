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
value_method = None
value_validation_method = None
detection = None

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
    case 2:
        detection = videoStream_TM_KNN.videoStream()
    case 3:
        detection = detect_CNN.ObjectDetection()
    case 4:
        detection = detect_CNN_KNN.ObjectDetection()

print('Choose validation method')
print('1: Execution Time')
print('2: Precision')

while True:
    value = input()
    try:
       value = int(value)
    except ValueError:
       print ('Invalid range, please enter: 1-4')
       continue
    if 1 <= value <= 2:
       break
    else:
       print ('Invalid range, please enter: 1-4')
  
def time_measure():
    detection.time_measure(r'WIN_20230522_15_59_16_Pro.mp4')
    return None

def precision_measure():
    return None  
     
match(value):
    case 1:
        time_measure()
    case 2:
        precision_measure()
        
        