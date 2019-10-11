import argparse
import cv2
from collections import Counter
from data.emotions import load_as_training_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--save')
    args = parser.parse_args()

    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)

    td = load_as_training_data(args.data)
    td.setTrainTestSplitRatio(0.8)

    print(Counter(td.getResponses()))

    svm.train(td.getTrainSamples(), cv2.ml.ROW_SAMPLE, td.getTrainResponses())

    print(td.getTestSamples().shape)
    print(td.getTestSamples().dtype)
    print(type(td.getTestSamples()))
    print(td.getTestSamples()[:1])
    print(svm.predict(td.getTestSamples()[:1]))

    res = svm.predict(td.getTestSamples())
    y_hat = res[1]
    correct = y_hat == td.getTestResponses()
    print(correct.sum() / len(y_hat))

    svm.train(td.getSamples(), cv2.ml.ROW_SAMPLE, td.getResponses())

    res = svm.predict(td.getSamples())
    y_hat = res[1]
    correct = y_hat.flatten() == td.getResponses().flatten()
    print(correct.sum() / len(y_hat))
    if args.save:
        svm.save(args.save)


    # TEST 
    # svm2 = cv2.ml.SVM_create()
    svm3 = cv2.ml.SVM_load(args.save)

    print(svm3.predict(td.getSamples()))
