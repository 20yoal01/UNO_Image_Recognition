import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MODEL = 'TM_KNN' + '_Result.csv'
PATH = 'result/bearbetad data/' + MODEL
OUTPUT = 'result/confusion_matrix/' + MODEL + '_cm'
arr = np.loadtxt(PATH, delimiter=";", dtype=str)

symbol = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'd2', 'r', 's', 'shuffle', 'card', 'custom', 'd4']
color = ['blue', 'green', 'red', 'yellow', 'wild']

symbol_true = arr[:,1]
symbol_pred = arr[:,3]

color_true = arr[:,0]
color_pred = arr[:,2]

print(color_pred)

def cm_symbol():
    cm = confusion_matrix(symbol_true, symbol_pred, labels=symbol)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=symbol)
    disp.plot()
    plt.show()
    
def cm_color():
    cm = confusion_matrix(color_true, color_pred, labels=color)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=color)
    disp.plot()
    plt.show()
    
cm_symbol()