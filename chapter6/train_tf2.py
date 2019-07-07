import tensorflow as tf
from datasets import gtsrb
import numpy as np
import cv2


def normalize(x):
    one_size = cv2.resize(x, (32, 32)).astype(np.float32) / 255
    return one_size - np.mean(one_size)


def train_tf_model(X_train, y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(44, (3, 3), input_shape=(32, 32, 3),
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(21, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10)
    return model


if __name__ == '__main__':
    (raw_X_train, y_train), (raw_X_test, y_test) = gtsrb.load_data()
    X_train = np.array([normalize(x) for x in raw_X_train])
    X_test = np.array([normalize(x) for x in raw_X_test])
    model = train_tf_model(X_train, y_train)
    tf.saved_model.save(model, "models/10/")
