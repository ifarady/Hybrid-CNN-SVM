
# the original code is from https://github.com/snatch59/cnn-svm-classifier.git
# I modified the code to fit my project. All credit goes to the original author

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.platform import gfile
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re
# -*- coding: utf-8 -*-


# Define the training and test data sets directories
model_dir = 'pretrained'
images_dir = 'tile-train/'

# TensorFlow inception-v3 feature extraction parameters
def create_graph(model_dir):
    """Create the CNN graph"""
    with tf.io.gfile.GFile(os.path.join(model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extract_features(list_images):
    """Extract bottleneck features"""
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    labels = []
    create_graph()
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image.
    with tf.compat.v1.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for ind, image in enumerate(list_images):
            imlabel = image.split('/')[1]
            # rough indication of progress
            if ind % 100 == 0:
                print('Processing', image, imlabel)
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)
            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
            labels.append(imlabel)
    return features, labels

# Graphics
def plot_features(feature_labels, t_sne_features):
    """feature plot"""
    plt.figure(figsize=(9, 9), dpi=100)
    uniques = {x: labels.count(x) for x in feature_labels}
    od = collections.OrderedDict(sorted(uniques.items()))
    colors = itertools.cycle(["r", "b", "g", "c", "m", "y",
                              "slategray", "plum", "cornflowerblue",
                              "hotpink", "darkorange", "forestgreen",
                              "tan", "firebrick", "sandybrown"])
    n = 0
    for label in od:
        count = od[label]
        m = n + count
        plt.scatter(t_sne_features[n:m, 0], t_sne_features[n:m, 1], c=next(colors), s=10, edgecolors='none')
        c = (m + n) // 2
        plt.annotate(label, (t_sne_features[c, 0], t_sne_features[c, 1]))
        n = m
    plt.show()

def plot_confusion_matrix(y_true, y_pred, matrix_title):
    """confusion matrix computation and display"""
    plt.figure(figsize=(7, 7), dpi=80)
    # use sklearn confusion matrix
    cm_array = confusion_matrix(y_true, y_pred)
    plt.imshow(cm_array[:-1, :-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(matrix_title, fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks, pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tight_layout()
    plt.show()
print ()

# Classifier performance
def run_classifier(clfr, x_train_data, y_train_data, x_test_data, y_test_data, acc_str, matrix_header_str):
    """run chosen classifier and display results"""
    start_time = time.time()
    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)
    print("%f seconds" % (time.time() - start_time))
    # confusion matrix computation and display
    print(acc_str.format(accuracy_score(y_test_data, y_pred) * 100))
    print(matrix_header_str)
    plot_confusion_matrix(y_test_data, y_pred, matrix_header_str)

# Read in images and extract features
# get images - labels are from the subdirectory names
if os.path.exists('features'):
    print('Pre-extracted features and labels found. Loading them ...')
    features = pickle.load(open('features', 'rb'))
    labels = pickle.load(open('labels', 'rb'))
else:
    print('No pre-extracted features - extracting features ...')
    # get the images and the labels from the sub-directory names
    dir_list = [x[0] for x in os.walk(images_dir)]
    dir_list = dir_list[1:]
    list_images = []
    for image_sub_dir in dir_list:
        sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('jpg|JPG', f)]
        list_images.extend(sub_dir_images)
    # extract features
    features, labels = extract_features(list_images)
    # save, so they can be used without re-running the last step which can be quite long
    pickle.dump(features, open('features', 'wb'))
    pickle.dump(labels, open('labels', 'wb'))
    print('CNN features obtained and saved.')

# Classification
# TSNE defaults:
# n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
# n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0,
# random_state=None, method=’barnes_hut’, angle=0.5
# t-sne feature plot
if os.path.exists('tsne_features.npz'):
    print('t-sne features found. Loading ...')
    tsne_features = np.load('tsne_features.npz')['tsne_features']
else:
    print('No t-sne features found. Obtaining ...')
    tsne_features = TSNE(perplexity=30.0).fit_transform(features)
    np.savez('tsne_features', tsne_features=tsne_features)
    print('t-sne features obtained and saved.')
plot_features(labels, tsne_features)

# prepare training and test datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)
# X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)

# LinearSVC defaults:
# penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, multi_class=’ovr’, fit_intercept=True,
# intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000
# classify the images with a Linear Support Vector Machine (SVM)
print('Support Vector Machine starting ...')
clf = LinearSVC()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-SVM Accuracy: {0:0.1f}%", "SVM Confusion matrix")
# print (features)
# print (len(features))
# print (features[0])
# print (len(features[0]))
# print (labels)
# print (len(labels))

# np.savetxt("ftr_test-cifar10-blur.csv", features, delimiter =",", fmt ='% s')
# np.savetxt("lbl_test-cifar10-blur.csv", labels, delimiter =",", fmt ='% s')
print ("================================")
# cnf_matrix = confusion_matrix(pred_labels, y_pred)
# print(cnf_matrix)

