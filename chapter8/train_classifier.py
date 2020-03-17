import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from data.store import load_collected_data
from data.process import train_test_split
from data.process import pca_featurize, _pca_featurize
from data.process import one_hot_encode
from data.store import pickle_dump


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--save', type=Path)
    parser.add_argument('--num-components', type=int,
                        default=20)
    args = parser.parse_args()

    data, targets = load_collected_data(args.data)

    train, test = train_test_split(len(data), 0.8)
    x_train, pca_args = pca_featurize(np.array(data)[train],
                                      num_components=args.num_components)

    encoded_targets, index_to_label = one_hot_encode(targets)

    last_layer_count = len(encoded_targets[0])
    mlp = cv2.ml.ANN_MLP_create()
    mlp.setLayerSizes(np.array([args.num_components, 10, last_layer_count], dtype=np.uint8))
    mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.1)
    mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    mlp.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.000001 ))

    y_train = encoded_targets[train]

    mlp.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

    x_test = _pca_featurize(np.array(data)[test], *pca_args)
    _, predicted = mlp.predict(x_test)

    y_hat = np.array([index_to_label[np.argmax(y)] for y in predicted])
    y_true = np.array(targets)[test]

    print('Training Accuracy:')
    print(sum(y_hat == y_true) / len(y_hat))

    if args.save:
        x_all, pca_args = pca_featurize(np.array(data), num_components=args.num_components)
        mlp.train(x_all, cv2.ml.ROW_SAMPLE, encoded_targets)
        args.save.mkdir(exist_ok=True)
        mlp.save(str(args.save / 'mlp.xml'))
        pickle_dump(index_to_label, args.save / 'index_to_label')
        pickle_dump(pca_args, args.save / 'pca_args')
