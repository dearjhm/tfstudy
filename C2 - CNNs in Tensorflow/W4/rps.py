# Import libraries
import tensorflow as tf
import os
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

RPS_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip'
pathTozip = tf.keras.utils.get_file('rps', origin=RPS_URL, extract=True)
basedir_rps = os.path.join(os.path.dirname(pathTozip), 'rps')

RPS_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip'
pathTozip = tf.keras.utils.get_file('rps-test-set', origin=RPS_URL, extract=True)
basedir_rpstest = os.path.join(os.path.dirname(pathTozip), 'rps-test-set')

#import os
#import zipfile
#
#local_zip = 'rps.zip'
#zip_ref = zipfile.ZipFile(local_zip, 'r')
#zip_ref.extractall()
#zip_ref.close()
#
#local_zip = 'rps-test-set.zip'
#zip_ref = zipfile.ZipFile(local_zip, 'r')
#zip_ref.extractall('')
#zip_ref.close()


# Training directory and datagenerator
TRAINING_DIR = "rps"
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")

# Validation directory and datagenerator
VALIDATION_DIR = "rps-test-set"
validation_data = ImageDataGenerator(rescale=1./255)

# Training generator
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    class_mode="categorical",
    batch_size=126
)

validation_generator = training_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    class_mode="categorical",
    batch_size=126)

# Create model
model = tf.keras.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # THis is the furst convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=15, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()