import pandas as pd
from gensim.models import Word2Vec
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
import keras.backend as K

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)



print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews"%(train["review"].size,test["review"].size,unlabeled_train["review"].size))





def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    words = review_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)




tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if(len(raw_sentence)>0):
            sentences.append(review_to_wordlist(raw_sentence,remove_stopwords))
    return sentences


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3
vector_size = 300
model = Word2Vec.load("300features_40minwords_10context")

# print(model["flower"])

# model.doesnt_match("france england germany berlin".split())
# model.doesnt_match("paris berlin london austria".split())
# model.most_similar("man")
# model.most_similar("queen")
# model.most_similar("awful")

def makeFeatureVec(words, model, num_features):
	featureVec = np.zeros((num_features,),dtype="float32")

	nwords = 0
	index2word_set = set(model.wv.index2word)

	for word in words:
		if word in index2word_set:
			nwords = nwords + 1
			featureVec = np.add(featureVec,model[word])

	featureVec = np.divide(featureVec,nwords)
	return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
	counter = 0
	reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
	#X_train = np.zeros((train_size, max_tweet_length, vector_size), dtype=K.floatx())

	for review in reviews:
		if(counter%1000==0):
			print("Review %d of %d" % (counter, len(reviews)))

		reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
		counter = counter + 1
	return reviewFeatureVecs


##########################################################################################

def getFeatures(words, model, num_features, max_review_length):
    featureVec = np.zeros((max_review_length, num_features,),dtype=K.floatx())
    nwords = 0
    index2word_set = set(model.wv.index2word)
    for word in words:
        if nwords>=250:
            break
        if word in index2word_set:
            featureVec[nwords] = model[word]
            nwords = nwords + 1
    return featureVec




def getFeatureVecs(reviews, model, num_features):
	max_review_length = 250
	X_train = np.zeros((len(reviews), max_review_length, num_features))
	
	for i, review in enumerate(reviews):
		if(i%1000==0):
			print("Review %d of %d" % (i, len(reviews)))
		X_train[i] = getFeatures(review, model, num_features, max_review_length)

	return X_train


##################################################################
print("Creating average feature vecs for train reviews..")

clean_train_reviews = []
for review in train["review"]:
	clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

#trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

X_train = getFeatureVecs(clean_train_reviews[1:2000], model, num_features)
Y_train = train["sentiment"]
y_train_sub = Y_train[1:2000] 
X_test = getFeatureVecs(clean_train_reviews[2001:3000], model, num_features)

# print("Creating average feature vecs for test reviews..")
# clean_test_reviews = []
# for review in test["review"]:
# 	clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

#testDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
#X_test = getFeatureVecs(clean_test_reviews, model, num_features)




##########

# from sklearn.ensemble import RandomForestClassifier
# forest = RandomForestClassifier(n_estimators=100)

# print("Fitting a random forest to labeled training data...")
# forest = forest.fit(trainDataVecs,train["sentiment"])

# Y_train_predict = forest.predict(trainDataVecs)
# from sklearn.metrics import accuracy_score
# accuracy_score(train["sentiment"],Y_train_predict)

# result = forest.predict(testDataVecs)



# Using embedding from Keras
#embedding_vecor_length = 300
vector_size = 300
max_review_length = 250
print("Training start........")
model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

# Convolutional model (3x conv, flatten, 2x dense)
model.add(Convolution1D(64, 3, padding='same', input_shape=(max_review_length, vector_size)))
model.add(Convolution1D(32, 3, padding='same'))
model.add(Convolution1D(16, 3, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(180,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# Log to tensorboard
tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_sub, epochs=1, callbacks=[tensorBoardCallback], batch_size=128)

# Evaluation on the test set
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
result = model.predict(X_test)
result[result>0.5] = 1
result[result<0.5] = 0
#output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
#output.to_csv("Word2Vec_AverageVectors1.csv", index=False, quoting=3)


