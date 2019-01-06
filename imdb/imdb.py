# IMDB dataset

import keras
from keras import layers, models
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

max_features = 200 # number of words to consider as features
max_len = 500 # cuts off text after this number of words, among max_features most common words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:v for k,v in word_to_id.items()}
id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in x_train[0]))

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

model = models.Sequential()

model.add(layers.Embedding(max_features, 128,
                           input_length=max_len,
                           name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# add keras callback to tensorboard
from datetime import datetime
now = datetime.now().strftime("%Y%m%d%H%M%S")
root_logdir = "C:/Temp/tensorboard"
logdir = "{}/run-{}/".format(root_logdir, now)

callbacks = [
        keras.callbacks.TensorBoard(
                log_dir=logdir,
                histogram_freq=1,
                embeddings_freq=1,
                embeddings_data = np.arange(0, max_len).reshape((1, max_len))
                )
]

# train
history = model.fit(x_train, y_train,
                    epochs=20,
                    batch_size=128,
                    validation_split=0.2,
                    callbacks=callbacks)

# start tensorboard: 
# tensorboard --logdir=C:\Temp\tensorboard\
