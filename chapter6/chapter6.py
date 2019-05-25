#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""OpenCV with Python Blueprints
    Chapter 6: Learning to Recognize Traffic Signs

    Traffic sign recognition using support vector machines (SVMs).
    SVMs are extended for multi-class classification using the "one-vs-one"
    and "one-vs-all" strategies.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from datasets import gtsrb
# from classifiers import MultiClassSVM
from classifiers import OneVsAllMultiSVM

__author__ = "Michael Beyeler"
__license__ = "GNU GPL 3.0 or later"


def main():
    strategies = ['one-vs-one', 'one-vs-all']
    # features = [None, 'gray', 'rgb', 'hsv', 'hog']
    features = ['hog']
    accuracy = np.zeros((2, len(features)))
    precision = np.zeros((2, len(features)))
    recall = np.zeros((2, len(features)))

    for i, feature in enumerate(features):
        print("feature", feature)
        (X_train, y_train), (X_test, y_test) = gtsrb.load_data(
            "datasets/gtsrb_training",
            feature=feature,
            test_split=0.2,
            seed=42)

        # convert to numpy
        X_train = np.squeeze(np.array(X_train)).astype(np.float32)
        y_train = np.array(y_train).astype(np.int)
        X_test = np.squeeze(np.array(X_test)).astype(np.float32)
        y_test = np.array(y_test).astype(np.int)

        print(X_train.shape, y_train.shape)
        print(X_train.dtype, y_train.dtype)
        print(min(X_train[0]), max(X_train[0]))
        print(X_train[0])

        # find all class labels
        unique_labels = np.unique(np.hstack((y_train, y_test)))
        print(f'number of unique labels is {len(unique_labels)}')

        # TODO: check that split has all the classes.

        single_svm = cv2.ml.SVM_create()
        single_svm.setKernel(cv2.ml.SVM_LINEAR)
        single_svm.setType(cv2.ml.SVM_C_SVC)
        single_svm.setC(2.67)
        single_svm.setGamma(5.383)
        single_svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        res = single_svm.predict(X_test)
        print('res', res)
        y_predict = res[1].flatten()
        mask = y_predict == y_test
        correct = np.count_nonzero(mask)
        print('correct', correct)
        print(100 * correct / y_predict.size)

        accuracy[0, i] = correct / y_predict.size
        accuracy[1, i] = correct / y_predict.size
        precision[0, i] = correct / y_predict.size
        precision[1, i] = correct / y_predict.size
        recall[0, i] = correct / y_predict.size
        recall[1, i] = correct / y_predict.size
        
        # one_vs_all_svm = OneVsAllMultiSVM()

        # one_vs_all_svm.train(X_train, y_train)

        # y_pred = one_vs_all_svm.predict_all(X_test)

        # for s_i, strategy in enumerate(strategies):
        #     # set up SVMs

        #     # training phase
        #     print(f"{strategy} - train")
        #     MCS.train(X_train, y_train)

        #     # test phase
        #     print(f"{strategy} - test")
        #     y_pred = MCS.predict_all(X_test)
        #     # acc, prec, rec = MCS.evaluate(X_test, y_test)
        #     accuracy[s, f] = acc
        #     precision[s, f] = np.mean(prec)
        #     recall[s, f] = np.mean(rec)
        #     print("       - accuracy: ", acc)
        #     print("       - mean precision: ", np.mean(prec))
        #     print("       - mean recall: ", np.mean(rec))

    # plot results as stacked bar plot
    f, ax = plt.subplots(2)
    for s in range(len(strategies)):
        x = np.arange(len(features))
        ax[s].bar(x - 0.2, accuracy[s, :], width=0.2, color='b',
                  hatch='/', align='center')
        ax[s].bar(x, precision[s, :], width=0.2, color='r', hatch='\\',
                  align='center')
        ax[s].bar(x + 0.2, recall[s, :], width=0.2, color='g', hatch='x',
                  align='center')
        ax[s].axis([-0.5, len(features) + 0.5, 0, 1.5])
        ax[s].legend(('Accuracy', 'Precision', 'Recall'), loc=2, ncol=3,
                     mode='expand')
        ax[s].set_xticks(np.arange(len(features)))
        ax[s].set_xticklabels(features)
        ax[s].set_title(strategies[s])

    plt.show()


if __name__ == '__main__':
    main()
