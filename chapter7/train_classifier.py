import argparse
import cv2

from data.emotions import load_as_training_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()

    training_data = load_as_training_data(args.data)

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(training_data)

    res = svm.predict(training_data.getSamples())
    y_hat = res[1].flatten()
    print(y_hat)
    print(sum(y_hat == training_data.getResponses()))


