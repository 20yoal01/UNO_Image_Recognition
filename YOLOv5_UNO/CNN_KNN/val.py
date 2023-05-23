import os
import cv2 as cv
import numpy as np
import detect_CNN_KNN


PATH = 'validation bilder/'
file_dir = os.listdir('validation bilder')
result = []
total_files = 0
correct = 0

TP = 0
TN = 0
FN = 0
FP = 0

for file in file_dir:
    total_files += 1
    cardlabel = [file[:file.find('_')], file[file.find('_')+1:len(file)-6]]
    color = cardlabel[0]
    symbol = cardlabel[1]
    print('-------------------')
    print("Färg: ", color)
    print("Kort: ", symbol)        
    
    matchedCard = detect_CNN_KNN.process_image(PATH + file)
    predicted_card_label = str(matchedCard[0]).split(' ', 1)
    foundCard = bool(matchedCard[1])
    
    # True Negative
    if foundCard is False and color == 'noise':
        correct += 1
        TN += 1
        print('TN')
        result.append(['facit:' + str(cardlabel), 'prediction:', str(predicted_card_label)])
        continue
    
    # False Negative
    if foundCard is False and color != 'noise':
        FN += 1
        print('FN')
        result.append(['facit:' + str(cardlabel), 'prediction:', str(predicted_card_label)])
        continue
    
    # True Positive
    if (''.join(cardlabel)).lower() == (''.join(predicted_card_label)).lower():
        TP += 1
        correct += 1
        print('MATCH')
    else:
        FP += 1
        print('FP')
    
    
    
    result.append(['facit:' + str(cardlabel), 'prediction:', str(predicted_card_label)])
    print('-------------------')
    

accuray = correct / total_files
precision = TP / (TP + FP)
recall = TP / (TP + FN)

metrics = ['accuracy: ' + str(accuray), 'precision: ' + str(precision), 'recall: ' + str(recall)]
arr_metrics = np.array(metrics)

print('FP:', FP)

print('Accuracy', str(correct) + '/' + str(total_files), accuray)
print('Precision', precision)
print('Recall', recall)

arr_result = np.array(result)
np.savetxt('result/CNN_KNN_Result.csv', arr_result, fmt="%s", delimiter=",")
np.savetxt('result/CNN_KNN_Metrics.csv', arr_metrics, fmt="%s", delimiter=",")

