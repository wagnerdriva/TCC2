import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.vgg16 import VGG16

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

image_size = (224, 224)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/train",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "Data/train",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical"
)

NUM_CLASSES = len(train_ds.class_names)
print(train_ds.class_names, NUM_CLASSES)

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

 # Use pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_size + (3,))

# Allow parameter updates for all layers
base_model.trainable = True

add_to_base = base_model.output
add_to_base = Flatten(name='head_flatten')(add_to_base)
add_to_base = Dense(1024, activation='relu', name='head_fc_1')(add_to_base)
add_to_base = Dropout(0.3, name='head_drop_1')(add_to_base)
add_to_base = Dense(1024, activation='relu', name='head_fc_2')(add_to_base)
add_to_base = Dropout(0.3, name='head_drop_2')(add_to_base)

# Add new output layer as head
new_output = Dense(NUM_CLASSES, activation='softmax', name='head_pred')(add_to_base)

# Define model
model = Model(base_model.input, new_output)

model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',metrics=['categorical_accuracy', 'accuracy'])

# Define early stopping as callback
early_stopping = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=12,
                                restore_best_weights=True)

checkpoint = ModelCheckpoint("models/VGG16-Colors-{epoch:02d}-{val_loss:.2f}-{val_categorical_accuracy:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

callbacks = [early_stopping, checkpoint]

# Fitting
history = model.fit(train_ds,
                    epochs=20,
                    validation_data=val_ds,
                    callbacks=callbacks)


model.save("models/finalVGG16-Colors.hdf5")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('images/LossVal_lossVGG16-Colors.png')

plt.clf()

plt.plot(history.history['categorical_accuracy'], label='train acc')
plt.plot(history.history['val_categorical_accuracy'], label='val acc')
plt.legend()
plt.savefig('images/AccVal_accVGG16-Colors.png')



