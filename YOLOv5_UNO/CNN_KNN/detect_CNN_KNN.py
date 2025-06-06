import torch
import numpy as np
import cv2 as cv
import time
#import YOLOv5_UNO.CNN_KNN.knn as knn
import knn 

class ObjectDetection:

    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        # self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.device = 'cuda'
        print('\n\nDevice Used: ', self.device)
        
    
    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'YOLOv5_UNO/runs/train/yolo_uno_det_low_v2/weights/best.pt')
        return model
    
    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame):
        cardName = ''
        labels, cord = results
        n = len(labels)
        print(n)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                #cv.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
                cropped_image = frame[y1:y2, x1:x2]
                #cv.imshow('cr', cropped_image)
                #cv.waitKey(0)
                
                print(self.class_to_label(labels[i])[0:4])
                if (self.class_to_label(labels[i]))[0:4] != 'wild' and (self.class_to_label(labels[i]))[0:2] !='d4':
                    color = knn.getColor(cropped_image)
                    cardName = color + ' ' + self.class_to_label(labels[i])
                else:
                    cardName = self.class_to_label(labels[i])
                    cardName = str(cardName).replace('_', ' ')
                print(cardName)
                cv.putText(frame, cardName, (x1, y1), cv.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 1)
                
        return frame, cardName
    
    def __call__(self):
        cap = cv.VideoCapture(0)
        
        while cap.isOpened():
            
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = self.score_frame(rgb)
            frame, cardName = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 1)
            cv.imshow("img", frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            
    def time_measure(self, path):
        cap = cv.VideoCapture(path)
        video_lengh = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        time_list = []
        
        frameCount = 0
        
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        
        print(video_lengh)
        
        for i in range(video_lengh):
            
            frameCount += 1
            
            if frameCount > 30:
                frameCount = 0
                
                start_time = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                results = self.score_frame(gray)
                frame, cardName = self.plot_boxes(results, frame)
                end_time = time.perf_counter()
                time_ms = round(((end_time - start_time) / 1000), 5)
                time_list.append(time_ms)
                fps = 1 / np.round(end_time - start_time, 3)
                #cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 1)
                #cv.imshow("img", frame)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        average_time = sum(time_list) / len(time_list)
        print(average_time)

def resize(img, scale_percent=.05):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

def process_image(path):
    img = cv.imread(path)
    img = resize(img, scale_percent=1.25)
    detection = ObjectDetection()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    results = detection.score_frame(gray)
    
    labels, cord = results
    n = len(labels)
    
    img, cardName = detection.plot_boxes(results, img)
    
    print('return: ' + cardName)
    return cardName, n
    #cv.imshow('Detected', img)
    #cv.waitKey(0)
    
#detect = ObjectDetection()
#detect()

process_image(r'validation bilder/WILD_card_4.jpg')
cv.waitKey(0)
