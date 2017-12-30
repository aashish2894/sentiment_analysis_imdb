import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import logging

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
import keras.backend as K
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

import os
import sys

from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from keras import optimizers


train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

print("Read %d labeled train reviews, %d labeled test reviews"%(train["review"].size,test["review"].size))


def review_to_wordlist(review, remove_stopwords=False):
    ps = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    words = review_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        words = [ps.stem(w) for w in words]
        #words = [lancaster_stemmer.stem(w) for w in words]
    text = ' '.join(words)
    return(text)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

BASE_DIR = ''
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
#TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1600
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2


##################################################################
print("Creating average feature vecs for train reviews..")

texts = []
for review in train["review"]:
	texts.append(review_to_wordlist(review, remove_stopwords=True))
	

Y_train = train["sentiment"]

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

#labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = Y_train
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]




embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


######################

print("Training start........")
model = Sequential()

model.add(embedding_layer)
# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same'))
model.add(MaxPooling1D(3))
model.add(Convolution1D(32, 3, padding='same'))
model.add(MaxPooling1D(3))
model.add(Convolution1D(16, 3, padding='same'))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)

adam = optimizers.Adam(lr=0.001)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=4, callbacks=[checkpointer], batch_size=64)

del data
del texts
del Y_train
del labels

print("Creating average feature vecs for test reviews..")
texts_test = []
for review in test["review"]:
	texts_test.append(review_to_wordlist(review, remove_stopwords=True))
	
sequences_test = tokenizer.texts_to_sequences(texts_test)

X_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

model.load_weights('weights.hdf5')
result = model.predict(X_test)
result[result>0.5] = 1
result[result<0.5] = 0
result = result.astype(int)
result = np.squeeze(result)
id_t = test["id"]
output = pd.DataFrame(data={"id":id_t, "sentiment":result})
output.to_csv("Word2Vec_CNNVectors_300_pool_epoch4_lemma.csv", index=False, quoting=3)


