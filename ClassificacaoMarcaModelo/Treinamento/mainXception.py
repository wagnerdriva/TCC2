import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

image_size = (299, 299)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/clean2",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/clean2",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
    ]
)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

model = Xception(
    include_top=True,
    weights=None,
    classes=7,
    classifier_activation="softmax"
)

trainableLayers = 0
for layer in model.layers:
    layer.trainable = True
    trainableLayers += 1

print(f"TotalLayers: {len(model.layers)} | TrainableLayers: {trainableLayers}")

epochs = 15

callbacks = [
    keras.callbacks.ModelCheckpoint("models/Xception-7Classes{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
]

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('images/LossVal_lossXception-7Classes.png')

plt.clf()

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('images/AccVal_accXception-7Classes.png')