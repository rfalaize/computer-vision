# dogs vs cats
# kaggle competition from 2013
# https://www.kaggle.com/c/dogs-vs-cats/data
# download the data using: kaggle competitions download -c dogs-vs-cats

#%% Create smaller dataset

import os, shutil
from keras.preprocessing.image import ImageDataGenerator

project_dir = 'C:/Users/rhome/github/computer-vision/dogsvscats/'
data_dir = project_dir +'data/'

train_dir = os.path.join(data_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
validation_dir = os.path.join(data_dir, 'validation')
os.makedirs(validation_dir, exist_ok=True)
test_dir = os.path.join(data_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

train_cats_dir = os.path.join(train_dir, 'cat')
os.makedirs(train_cats_dir, exist_ok=True)
train_dogs_dir = os.path.join(train_dir, 'dog')
os.makedirs(train_dogs_dir, exist_ok=True)

validation_cats_dir = os.path.join(validation_dir, 'cat')
os.makedirs(validation_cats_dir, exist_ok=True)
validation_dogs_dir = os.path.join(validation_dir, 'dog')
os.makedirs(validation_dogs_dir, exist_ok=True)

test_cats_dir = os.path.join(test_dir, 'cat')
os.makedirs(test_cats_dir, exist_ok=True)
test_dogs_dir = os.path.join(test_dir, 'dog')
os.makedirs(test_dogs_dir, exist_ok=True)

#%% Copy images

# create a smaller dataset with 3 subsets
# - 2000 images for training set
# - 1000 images for validation set
# - 1000 images for test set
# training set is balanced so classification accuracy is an adequate measure for success.

'''
for c in ['cat', 'dog']:
    for i in range(1000):
        src = data_dir + 'kaggle/train/{}.{}.jpg'.format(c, i)
        dst = data_dir + 'train/' + c + '/{}.{}.jpg'.format(c, i)
        shutil.copy(src, dst)
    for i in range(1000, 1500):
        src = data_dir + 'kaggle/train/{}.{}.jpg'.format(c, i)
        dst = data_dir + 'validation/' + c + '/{}.{}.jpg'.format(c, i)
        shutil.copy(src, dst)
    for i in range(1500, 2000):
        src = data_dir + 'kaggle/train/{}.{}.jpg'.format(c, i)
        dst = data_dir + 'test/' + c + '/{}.{}.jpg'.format(c, i)
        shutil.copy(src, dst)

# validation
print('total train cat images:', len(os.listdir(train_cats_dir)))
print('total train dog images:', len(os.listdir(train_dogs_dir)))    
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir))) 
print('total test cat images:', len(os.listdir(train_cats_dir)))
print('total test dog images:', len(os.listdir(train_dogs_dir))) 
'''

#%% Training

from keras import models, layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0001),
              metrics=['acc'])


# %% Model 1
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# generator will resize images to 150*150
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

'''
# inspect one of the generators
for data_batch, labels_batch in train_generator:
    print(data_batch.shape)
    print(labels_batch.shape)
    break
'''

history = model.fit_generator(
        train_generator, 
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

model.save(project_dir + 'model_small_1.h5')

#%% Model 2
# use data augmentation to generate more training data
# via a number of random transformations that yield believable-looking images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
 
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

history = model.fit_generator(
        train_generator, 
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)

model.save(project_dir + 'model_small_2.h5')

'''
from keras.preprocessing import image
fnames = [os.path.join(train_cats_dir, x) for x in os.listdir(train_cats_dir)]
img_path = fnames[3] # choose one image to augment

img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img) # converts to numpy array
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i%4 == 0:
        break
plt.show()
'''

#%% Model 3
# use pretrained VGG16 model

from keras import models, layers
from keras import optimizers
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# freeze convolutional layers not to update weights during training
print('trainable layers before freezing conv_base:', len(model.trainable_weights))
conv_base.trainable = False
print('trainable layers after freezing conv_base:', len(model.trainable_weights))

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
 
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit_generator(
        train_generator, 
        steps_per_epoch=100,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=50)

model.save(project_dir + 'model_CVV16_1.h5')


#%% plot the loss and accuracy during training
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) +1)
plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Loss')
plt.legend()
plt.figure()

plt.show()
