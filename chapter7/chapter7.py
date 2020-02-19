#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenCV with Python Blueprints
Chapter 7: Learning to Recognize Traffic Signs

Traffic sign recognition using support vector machines (SVMs).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data.gtsrb import load_training_data
from data.gtsrb import load_test_data
from data.process import surf_featurize, hog_featurize
from data.process import hsv_featurize, grayscale_featurize


def train_MLP(X_train, y_train):
    mlp = cv2.ml.ANN_MLP_create()
    mlp.setLayerSizes(np.array([784, 512, 512, 10]))
    mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2.5, 1.0)
    mlp.setTrainingMethod(cv2.ml.ANN_MLP.BACKPROP)
    mlp.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return mlp


def train_one_vs_all_SVM(X_train, y_train):
    single_svm = cv2.ml.SVM_create()
    single_svm.setKernel(cv2.ml.SVM_LINEAR)
    single_svm.setType(cv2.ml.SVM_C_SVC)
    single_svm.setC(2.67)
    single_svm.setGamma(5.383)
    single_svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return single_svm


def accuracy(y_predicted, y_true):
    return sum(y_predicted == y_true) / len(y_true)


def precision(y_predicted, y_true, positive_label):
    cm = confusion_matrix(y_predicted, y_true)
    true_positives = cm[positive_label, positive_label]
    total_positives = sum(cm[positive_label])
    return true_positives / total_positives


def recall(y_predicted, y_true, positive_label):
    cm = confusion_matrix(y_predicted, y_true)
    true_positives = cm[positive_label, positive_label]
    class_members = sum(cm[:, positive_label])
    return true_positives / class_members


def confusion_matrix(y_predicted, y_true):
    num_classes = max(max(y_predicted), max(y_true)) + 1
    conf_matrix = np.zeros((num_classes, num_classes))
    for r, c in zip(y_predicted, y_true):
        conf_matrix[r, c] += 1
    return conf_matrix


def train_sklearn_random_forest(X_train, y_train):
    pass


def main(labels=[0, 10, 20, 30, 40]):
    train_data, train_labels = load_training_data(labels)
    test_data, test_labels = load_test_data(labels)

    y_train = np.array(train_labels)
    y_test = np.array(test_labels)

    accuracies = {}

    for featurize in [hog_featurize, grayscale_featurize,
                      hsv_featurize, surf_featurize]:
        x_train = featurize(train_data)
        print(x_train.shape)
        model = train_one_vs_all_SVM(x_train, y_train)

        x_test = featurize(test_data)
        res = model.predict(x_test)
        y_predict = res[1].flatten()
        np.save(f'y_predict_{featurize.__name__}', y_predict)
        np.save('y_true', y_test)
        accuracies[featurize.__name__] = accuracy(y_predict, y_test)

    print(accuracies)

    plt.bar(accuracies.keys(), accuracies.values())
    plt.axes().xaxis.set_tick_params(rotation=20)
    plt.ylim([0, 1])
    plt.grid()
    plt.title('Test accuracy for different featurize functions')
    plt.show()


if __name__ == '__main__':
    main()
