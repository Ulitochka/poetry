from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D, BatchNormalization
from gensim.models import KeyedVectors
from keras.callbacks import *
import numpy as np

from linguistic_style_transfer.linguistic_style_transfer_model.utils import data_processor

options = {
    "text_file_path": '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set/train.txt',
    "label_file_path": '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set/train_labels.txt',
    "vocab_size": 40000,
    "training_epochs": 100,
    "fastext": "/home/mdomrachev/Data/cc.ru.300.vec"
}

w2v_vectors = KeyedVectors.load_word2vec_format(options['fastext'], binary=False)

[word_index, x, _, _, _] = \
    data_processor.get_text_sequences(
        options['text_file_path'],
        options['vocab_size'],
        '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/author_identification/vocab')

vocabulary_index_sorted = sorted([(w, word_index[w]) for w in word_index], key= lambda x: x[1], reverse=False)
vectors = []
cover_voc = 0
base_vector = np.zeros(300)
for t in vocabulary_index_sorted:
    try:
        vectors.append(w2v_vectors[t[0]])
        cover_voc += 1
    except KeyError:
        vectors.append(base_vector)
vectors = np.array(vectors)
print('create matrix: %s; cover_voc: %s' % (vectors.shape, cover_voc), '\n')

x = np.asarray(x)

[y, _] = data_processor.get_labels(options['label_file_path'], store_labels=False, one_hot_encode=False)

shuffle_indices = np.random.permutation(np.arange(len(x)))
# shuffle_indices = [i for i in shuffle_indices if i != max(shuffle_indices)]

x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(0.01 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(options['vocab_size']))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

model = Sequential()
model.add(Embedding(input_dim=len(word_index),
                    output_dim=300,
                    input_length=15,
                    weights=[vectors],
                    mask_zero=False,
                    trainable=False))
model.add(Dropout(0.5))
model.add(Conv1D(filters=1024,
                 kernel_size=5,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.layers[1].trainable=False

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=15,
                               verbose=0,
                               mode='auto')

print('Train...')
model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          validation_data=(x_dev, y_dev),
          shuffle=True,
          callbacks=[early_stopping])
# score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
# print('Test score:', score)
# print('Test accuracy:', acc)


