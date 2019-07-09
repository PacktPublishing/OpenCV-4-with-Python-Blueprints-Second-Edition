import tensorflow as tf
from datasets import gtsrb
import numpy as np
import cv2
# https://answers.opencv.org/question/175699/readnetfromtensorflow-fails-on-retrained-nn/

# https://jeanvitor.com/tensorflow-object-detecion-opencv/

# https://heartbeat.fritz.ai/real-time-object-detection-on-raspberry-pi-using-opencv-dnn-98827255fa60

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


# if __name__ == '__main__':

(raw_X_train, y_train), (raw_X_test, y_test) = gtsrb.load_data()
X_train = np.array([normalize(x) for x in raw_X_train])
X_test = np.array([normalize(x) for x in raw_X_test])
model = train_tf_model(X_train, y_train)

model
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants
tf.saved_model.save(model, "models/10/")
builder = saved_model_builder.SavedModelBuilder("models/11/")
signature = predict_signature_def(inputs={"images": model.input},
                                      outputs={"scores": model.output})
with tf.keras.backend.get_session() as sess:
    # Save the meta graph and the variables
    builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                     signature_def_map={"predict": signature})
builder.save()
tf.train.write_graph(tf.keras.backend.get_session().graph_def,"./models",  "saved_model.pbtxt")


m = cv2.dnn.readNetFromTensorflow('models/11/saved_model.pb','models/saved_model.pbtxt')
m = cv2.dnn.readNetFromTensorflow('models/11/saved_model.pb')
