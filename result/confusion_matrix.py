import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import os

MODEL = 'TM_KNN' + '_Result.csv'
DIRECTORY = 'result/bearbetad data/'
PATH = DIRECTORY + MODEL
OUTPUT = 'result/confusion_matrix/'
OUTPUTF_FILE = OUTPUT

symbol = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'd2', 'r', 's', 'shuffle', 'card', 'custom', 'd4', 'background']
color = ['blue', 'green', 'red', 'yellow', 'wild', 'background']

arr = []
    
symbol_true = []
symbol_pred = []
color_true = []
color_pred = []

print(color_pred)


def cm_symbol():
    cm = confusion_matrix(symbol_true, symbol_pred, labels=symbol)
    cr = classification_report(symbol_true, symbol_pred, digits=3)
    print(cr)
    np.savetxt(OUTPUTF_FILE + '_' + 'symbol', [cr], fmt="%s", delimiter=",")
    #plot(cm, False)
    
def cm_color():
    cm = confusion_matrix(color_true, color_pred, labels=color)
    cr = classification_report(color_true, color_pred, digits=3)
    print(cr)
    np.savetxt(OUTPUTF_FILE + '_' + 'name', [cr], fmt="%s", delimiter=",")
    #plot(cm, True)

def plot(cm, isColor):
    import seaborn as sn
    labels = []
    
    if isColor:
        labels = color
    else:
        labels = symbol
    
    #cm[cm == 0] = np.nan  # don't annotate (would appear as 0.00)
    fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
    nc, nn = len(labels), len(labels)  # number of classes, names
    sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size


    sn.heatmap(cm,
               ax=ax,
               annot=nc < 30,
               annot_kws={
                   "size": 8},
               cmap='Blues',
               square=True,
               vmin=0.0,
               xticklabels=labels,
               yticklabels=labels).set_facecolor((1, 1, 1))
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    name = ''
    if isColor:
        name = 'color'
    else:
        name = 'symbol'
    fig.savefig(OUTPUTF_FILE + '_' + name, dpi=250)
    plt.close(fig)

def load_data():
    arr = np.loadtxt(PATH, delimiter=";", dtype=str)
    
    global symbol_true
    global symbol_pred
    global color_true
    global color_pred
     
    symbol_true = arr[:,1]
    symbol_pred = arr[:,3]

    color_true = arr[:,0]
    color_pred = arr[:,2]

for filename in os.listdir(DIRECTORY):
    f = os.path.join(DIRECTORY, filename)
    if os.path.isfile(f):
        MODEL = filename
        PATH = DIRECTORY + MODEL
        filename = filename[:len(filename)-4]
        OUTPUTF_FILE = OUTPUT + filename
        load_data()
        cm_symbol()
        cm_color()