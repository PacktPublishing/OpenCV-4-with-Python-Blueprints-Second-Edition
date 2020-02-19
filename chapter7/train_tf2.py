import tensorflow as tf
import numpy as np
import cv2
# https://answers.opencv.org/question/175699/readnetfromtensorflow-fails-on-retrained-nn/

# https://jeanvitor.com/tensorflow-object-detecion-opencv/

# https://heartbeat.fritz.ai/real-time-object-detection-on-raspberry-pi-using-opencv-dnn-98827255fa60

from data.gtsrb import load_training_data
from data.gtsrb import load_test_data

UNIFORM_SIZE = (32, 32)


def normalize(x):
    """
    Do minimum pre-processing
    1. resize to UNIFORM_SIZE
    2. scale to (0, 1) range
    3. subtract the mean of all pixel values
    """
    one_size = cv2.resize(x, UNIFORM_SIZE).astype(np.float32) / 255
    return one_size - one_size.mean()


def train_tf_model(X_train, y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(20, (8, 8),
                               input_shape=list(UNIFORM_SIZE) + [3],
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=4),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(43, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, np.array(train_labels), epochs=2)
    return model


if __name__ == '__main__':
    train_data, train_labels = load_training_data(labels=None)
    test_data, test_labels = load_test_data(labels=None)

    x_train = np.array([normalize(x) for x in train_data])
    model = train_tf_model(x_train, train_labels)
    x_test = np.array([normalize(x) for x in test_data])

    y_hat = model.predict_classes(x_test)

    acc = sum(y_hat == np.array(test_labels)) / len(test_labels)
    print(f'Accuracy = {acc:.3f}')
