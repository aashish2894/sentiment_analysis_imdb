import pandas as pd


train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)



print("Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews"%(train["review"].size,test["review"].size,unlabeled_train["review"].size))



from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    words = review_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)


import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if(len(raw_sentence)>0):
            sentences.append(review_to_wordlist(raw_sentence,remove_stopwords))
    return sentences


    
# sentences = []

# print("Parsing the sentences from the training set")

# for review in train["review"]:
#     sentences += review_to_sentences(review, tokenizer)

# print("Parsing the sentences from the unlabeled set")

# for review in unlabeled_train["review"]:
#     sentences += review_to_sentences(review, tokenizer)


    
# print(len(sentences))

# print(sentences[0])
# print(sentences[1])



import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3



from gensim.models import Word2Vec
model = Word2Vec.load("300features_40minwords_10context")

print(model["flower"])

model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())
model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")

import numpy as np

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

	for review in reviews:
		if(counter%1000==0):
			print("Review %d of %d" % (counter, len(reviews)))

		reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
		counter = counter + 1
	return reviewFeatureVecs


clean_train_reviews = []
for review in train["review"]:
	clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vecs for test reviews..")
clean_test_reviews = []
for review in test["review"]:
	clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

testDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)

print("Fitting a random forest to labeled training data...")
forest = forest.fit(trainDataVecs,train["sentiment"])

Y_train_predict = forest.predict(trainDataVecs)
from sklearn.metrics import accuracy_score
accuracy_score(train["sentiment"],Y_train_predict)

result = forest.predict(testDataVecs)

output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)


