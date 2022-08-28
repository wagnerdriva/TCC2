import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

image_size = (224, 224)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/HATCH",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/HATCH",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"
)

NUM_CLASSES = len(train_ds.class_names)

data_augmentation = keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ]
)

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

train_ds = augmented_train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

 # Use pre-trained InceptionV3 model
base_model = InceptionV3(weights=None, include_top=False, input_shape=image_size + (3,))

# Allow parameter updates for all layers
base_model.trainable = True

add_to_base = base_model.output
add_to_base = GlobalAveragePooling2D(data_format='channels_last', name='head_gap')(add_to_base)

# Add new output layer as head
new_output = Dense(NUM_CLASSES, activation='softmax', name='head_pred')(add_to_base)

# Define model
model = Model(base_model.input, new_output)

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy'])

# Define early stopping as callback
early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=12,
                                restore_best_weights=True)

checkpoint = ModelCheckpoint("models/InceptionV3-HATCH-{epoch:02d}-{val_loss:.2f}-{val_categorical_accuracy:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

callbacks = [early_stopping, checkpoint]

# Fitting
history = model.fit(train_ds,
                    epochs=30,
                    validation_data=val_ds,
                    callbacks=callbacks)


model.save("models/finalInceptionV3-HATCH-Test.hdf5")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('images/LossVal_lossInceptionV3-HATCH.png')

plt.clf()

plt.plot(history.history['categorical_accuracy'], label='train acc')
plt.plot(history.history['val_categorical_accuracy'], label='val acc')
plt.legend()
plt.savefig('images/AccVal_accInceptionV3-HATCH.png')
