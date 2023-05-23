import os
import cv2 as cv
import numpy as np
import card as card
import templateMatch as TM_templateMatch
#import TM_KNN.templateMatch as TM_KNN_templateMatch


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
    print('-------------------')
    print("Färg: ", cardlabel[0])
    print("Kort: ", cardlabel[1])
    
    img = cv.imread(PATH + file)

    procImg, cnt = card.process(img)
    print('cardImg length: ', len(procImg))
    
    if (len(procImg) > 1 and cardlabel[0] == 'noise') or (len(procImg) == 0 and cardlabel[0] == 'noise'):
        correct += 1
        TN += 1
        print('TN')
        result.append(['facit:' + str(cardlabel), 'prediction:', ''])
        continue
    elif len(procImg) > 1 and cardlabel[0] != 'noise' or (len(procImg) == 0 and cardlabel[0] != 'noise'):
        FN += 1   
        print('FN')
        result.append(['facit:' + str(cardlabel), 'prediction:', ''])
        continue
        
    for cardImg in procImg:
        #cv.imshow('a',cardImg)
        #cv.waitKey(0)
        matchedCard = TM_templateMatch.match(cardImg)
    predicted_card_label = str(matchedCard[0]).split(' ', 1)
    
    if 1 < len(predicted_card_label) and predicted_card_label[1].find('_upsidedown') != -1:
        symbol = predicted_card_label[1]
        corrected_symbol = symbol[:symbol.find('_upsidedown')]
        predicted_card_label[1] = corrected_symbol
        print('corrected: ' +  predicted_card_label[0] + ' ' + predicted_card_label[1])
    
    # True Negative
    if matchedCard[1] is False and cardlabel[0] == 'noise':
        correct += 1
        TN += 1
        print('TN')
        result.append(['facit:' + str(cardlabel), 'prediction:', str(predicted_card_label)])
        continue
    
    # False Negative
    if matchedCard[1] is False and cardlabel[0] != 'noise':
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
    
    #break

accuray = correct / total_files
precision = TP / (TP + FP)
recall = TP / (TP + FN)

metrics = ['accuracy: ' + str(accuray), 'precision: ' + str(precision), 'recall: ' + str(recall)]
arr_metrics = np.array(metrics)

print('Accuracy', str(correct) + '/' + str(total_files), accuray)
print('Precision', precision)
print('Recall', recall)

arr_result = np.array(result)
np.savetxt('result/TM_Result.csv', arr_result, fmt="%s", delimiter=",")
np.savetxt('result/TM_Metrics.csv', arr_metrics, fmt="%s", delimiter=",")

