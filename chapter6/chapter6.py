#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV with Python Blueprints
Chapter 6: Learning to Recognize Traffic Signs

Traffic sign recognition using support vector machines (SVMs).
SVMs are extended for multi-class classification using the "one-vs-one"
and "one-vs-all" strategies.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data.gtsrb import load_training_data
from data.gtsrb import load_test_data
from data.process import grayscale_featurize


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


def new_main(labels=[0, 10]):
    train_data, y_train = load_training_data(labels)
    test_data, y_test = load_test_data(labels)

    for featurize in [grayscale_featurize]:
        x_train = np.array(featurize(train_data))
        model = train_one_vs_all_SVM(x_train, y_train)

        x_test = featurize(test_data)
        res = model.predict(x_test)
        print('res', res)
        y_predict = res[1].flatten()
        mask = y_predict == y_test
        correct = np.count_nonzero(mask)
        print('correct', correct)
        print(100 * correct / y_predict.size)


def old_main():
    strategies = {
        'SVM': train_one_vs_all_SVM,
        'MLP': train_MLP,
    }
    # features = [None, 'gray', 'rgb', 'hsv', 'hog']
    features = ['rgb']

    accuracy = np.zeros((len(strategies), len(features)))
    precision = np.zeros((len(strategies), len(features)))
    recall = np.zeros((len(strategies), len(features)))

    for i_f, feature in enumerate(features):
        print("feature", feature)
        (X_train, y_train), (X_test, y_test) = gtsrb.load_data(
            "datasets/gtsrb_training",
            feature=feature,
            test_split=0.8,
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

        for i_s, (label, model_trainer) in enumerate(strategies.items()):
            model = model_trainer(X_train, y_train)
            res = model.predict(X_test)
            print('res', res)
            y_predict = res[1].flatten()
            mask = y_predict == y_test
            correct = np.count_nonzero(mask)
            print('correct', correct)
            print(100 * correct / y_predict.size)

            accuracy[i_s, i_f] = correct / y_predict.size
            precision[i_s, i_f] = correct / y_predict.size
            recall[i_s, i_f] = correct / y_predict.size
        
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

    # plot results as stacked bar plo[//
    # f, ax = plt.subplots(len(strategies))
    f, ax = plt.subplots(2)  # FIXME: train
    for i_s, (label, _) in enumerate(strategies.items()):
        x = np.arange(len(features))
        ax[i_s].bar(x - 0.2, accuracy[i_s, :],
                    width=0.2, color='b', hatch='/', align='center')
        ax[i_s].bar(x, precision[i_s, :],
                    width=0.2, color='r', hatch='\\', align='center')
        ax[i_s].bar(x + 0.2, recall[i_s, :],
                    width=0.2, color='g', hatch='x', align='center')
        ax[i_s].axis([-0.5, len(features) + 0.5, 0, 1.5])
        ax[i_s].legend(('Accuracy', 'Precision', 'Recall'),
                       loc=2, ncol=3, mode='expand')
        ax[i_s].set_xticks(np.arange(len(features)), features)
        ax[i_s].set_title(label)

    plt.show()


if __name__ == '__main__':
    main()
