from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.layers import Dense, Conv2D, Input, Dropout, MaxPool2D, ReLU, concatenate, AveragePooling2D 


import os
if not os.path.isdir('models'):
  os.mkdir('models')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"  # specify which GPU(s) to be used

image_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1./255., 
    rotation_range = 40, 
    width_shift_range = 0.2, 
    height_shift_range = 0.2, 
    shear_range = 0.2, 
    zoom_range = 0.2, 
    horizontal_flip = True,
    validation_split = 0.2
)

train_generator = train_datagen.flow_from_directory(
    "Data/CarV4", 
    batch_size = batch_size, 
    target_size = image_size,
    subset="training",
)

test_datagen = ImageDataGenerator(
    rescale = 1.0/255.,
    validation_split = 0.2,
)

validation_generator = test_datagen.flow_from_directory(
    "Data/CarV4", 
    batch_size = batch_size, 
    target_size = image_size,
    subset="validation",
)

print(train_generator.class_indices, len(train_generator.class_indices))


def fire_module(x,s1,e1,e3):
    s1x = Conv2D(s1,kernel_size = 1, padding = 'same')(x)
    s1x = ReLU()(s1x)
    e1x = Conv2D(e1,kernel_size = 1, padding = 'same')(s1x)
    e3x = Conv2D(e3,kernel_size = 3, padding = 'same')(s1x)
    x = concatenate([e1x,e3x])
    x = ReLU()(x)
    return x

def SqueezeNet(input_shape, nclasses):
    input = Input(input_shape)
    x = Conv2D(96,kernel_size=(7,7),strides=(2,2),padding='same')(input)
    x = MaxPool2D(pool_size=(3,3), strides = (2,2))(x)
    x = fire_module(x, s1 = 16, e1 = 64, e3 = 64) #2
    x = fire_module(x, s1 = 16, e1 = 64, e3 = 64) #3
    x = fire_module(x, s1 = 32, e1 = 128, e3 = 128) #4
    x = MaxPool2D(pool_size=(3,3), strides = (2,2))(x)
    x = fire_module(x, s1 = 32, e1 = 128, e3 = 128) #5
    x = fire_module(x, s1 = 48, e1 = 192, e3 = 192) #6
    x = fire_module(x, s1 = 48, e1 = 192, e3 = 192) #7
    x = fire_module(x, s1 = 64, e1 = 256, e3 = 256) #8
    x = MaxPool2D(pool_size=(3,3), strides = (2,2))(x)
    x = fire_module(x, s1 = 64, e1 = 256, e3 = 256) #9
    x = Dropout(0.5)(x)
    x = Conv2D(nclasses,kernel_size = 1)(x)
    output = AveragePooling2D(pool_size=(13,13))(x)
    model = Model(input, output)
    return model


model = SqueezeNet((32,), len(train_generator.class_indices))

checkpoint = ModelCheckpoint("models/SqueezeNet-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(
    train_generator,
    verbose=1,
    validation_data = validation_generator,
    epochs = 50,
    steps_per_epoch = len(train_generator),
    validation_steps = len(validation_generator),
    callbacks = [checkpoint],
)

model.save("models/finalSqueezeNet.hdf5")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('LossVal_lossSqueezeNet.png')

plt.clf()

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('AccVal_accSqueezeNet.png')


