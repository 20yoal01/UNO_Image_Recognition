import YOLOv5_UNO.detect_CNN_KNN as detect_CNN_KNN
import YOLOv5_UNO.detect_CNN as detect_CNN
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
        detection = detect_CNN_KNN.ObjectDetection()
        detection()
    case 2:
        detection = detect_CNN.ObjectDetection()
        detection()
    case 3:
           detection_CNN = detect_CNN_KNN.ObjectDetection()
           detection_KNN = detect_CNN.ObjectDetection()
           t1 = threading.Thread(target=detection_CNN)
           t2 = threading.Thread(target=detection_KNN)
           t1.start()
           t2.start()
           t1.join()
           t2.join()
