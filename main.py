import YOLOv5_UNO.detect as detect


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
        detection = detect.ObjectDetection()
        detection()
