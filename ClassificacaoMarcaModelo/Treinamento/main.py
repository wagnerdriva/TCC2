from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


import os
if not os.path.isdir('models'):
  os.mkdir('models')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1./255., 
    # rotation_range = 40, 
    # width_shift_range = 0.2, 
    # height_shift_range = 0.2, 
    # shear_range = 0.2, 
    # zoom_range = 0.2, 
    # horizontal_flip = True,
    validation_split = 0.2
)

train_generator = train_datagen.flow_from_directory(
    "Data/clean2", 
    batch_size = batch_size, 
    target_size = image_size,
    subset="training",
)

test_datagen = ImageDataGenerator(
    rescale = 1.0/255.,
    validation_split = 0.2,
)

validation_generator = test_datagen.flow_from_directory(
    "Data/clean2", 
    batch_size = batch_size, 
    target_size = image_size,
    subset="validation",
)

print(train_generator.class_indices, len(train_generator.class_indices))

IMAGE_SIZE = 224
base_model = InceptionV3(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

trainableLayers = 0
# don't train existing weights
for layer in model.layers[:279]:
    layer.trainable = False
for layer in model.layers[279:]:
    layer.trainable = True
    trainableLayers += 1

print(f"TotalLayers: {len(model.layers)} | TrainableLayers: {trainableLayers}")

checkpoint = ModelCheckpoint("models/InceptionV3-7Classes-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(
    train_generator,
    verbose=1,
    validation_data = validation_generator,
    epochs = 30,
    steps_per_epoch = len(train_generator),
    validation_steps = len(validation_generator),
    callbacks = [checkpoint],
)

model.save("models/finalInceptionV3-7Classes.hdf5")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_lossInceptionV3-7Classes.png')
plt.savefig('images/LossVal_lossInceptionV3-7Classes.png')

plt.clf()

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_accInceptionV3-7Classes.png')
plt.savefig('images/AccVal_accInceptionV3-7Classes.png')



