import tensorflow.keras as K

from data import ds

base_model = K.applications.MobileNetV2(input_shape=(224,224, 3), include_top=False)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

x = K.layers.GlobalAveragePooling2D()(base_model.output)

is_breeds = True
if is_breeds:
    out = K.layers.Dense(37,activation="softmax")(x)
    inp_ds = ds.map(lambda d: (d.image,d.breed))
else:
    out = K.layers.Dense(2,activation="softmax")(x)
    inp_ds = ds.map(lambda d: (d.image,d.type))

model = K.Model(inputs=base_model.input, outputs=out)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy","top_k_categorical_accuracy"])

valid = inp_ds.take(1000)
train = inp_ds.skip(1000).shuffle(10**4)

model.fit(train.batch(32), epochs=4)
model.evaluate(valid.batch(1))
