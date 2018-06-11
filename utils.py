import imageio
import numpy as np
from tqdm import tqdm
import os
import itertools
import matplotlib.pyplot as plt


classes = ['art_painting', 'cartoon', 'photo', 'sketch']
types = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

def read_images_from_folder(dir_path):
    files = os.listdir(dir_path)
    images = []
    for f in files:
        images.append(imageio.imread(os.path.join(dir_path, f)))
    return images

def read_data_pcas(data_path='PCAS_DATASET', test_type='giraffe'):
    assert test_type in types
    
    X_train, y_train = [], []
    X_test, y_test = [], []

    for c in tqdm(classes, total = len(classes)):
        cur_class_images = []
        for t in types:
            dir_path = os.path.join(data_path, '%s/%s' % (c, t))
            dir_images = read_images_from_folder(dir_path)
            if t != test_type:
                X_train.extend(dir_images)
                y_train.extend([c] * len(dir_images))
            else:
                X_test.extend(dir_images)
                y_test.extend([c] * len(dir_images))

    X_train = np.stack(X_train)
    y_train = np.array(y_train)
    X_test = np.stack(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

# Code from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



0.23, 0.21, 0.87, 0.999, 0.977, 0.98, 0.983, 0.978
0.23, 0.20, 0.75, 0.988, 0.905, 0.92, 0.965, 0.95
0.16, 0.16, 0.73, 0.864, 0.864, 0.82, 0.847, 0.80