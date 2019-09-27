import tensorflow as tf
import tensorflow.contrib.keras as K

model = K.models.Sequential()

model.add(
    K.layers.Conv2D(
        filters=5,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        input_shape=(6, 7, 3)
        )
    )
model.add( K.layers.Reshape((6*7*5,)) )
model.add(
    K.layers.Dense(
        units=20,
        activation=K.activations.softmax)
    )
model.add(
    K.layers.Dense(
        units=10,
        activation=K.activations.softmax)
    )
from model_utils import save_model
import cv2
save_model(model,"custom.pb")
net = cv2.dnn.readNetFromTensorflow('custom.pb')
