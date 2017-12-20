#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 19:42:48 2017

@author: aashish
"""

import pandas as pd
train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

from bs4 import BeautifulSoup

example1 = BeautifulSoup(train["review"][0])

#print(train["review"][0])
#print(example1.get_text())

import re

letters_only = re.sub("[^a-zA-z]", " ", example1.get_text())

#print(letters_only)

lower_case = letters_only.lower()
words = lower_case.split()

import nltk
from nltk.corpus import stopwords

#print(stopwords.words("english"))
#print(len(words))
words = [w for w in words if not w in stopwords.words("english")]
#print(len(words))


def review_to_words(raw_review):
    #remove html 
    review_text = BeautifulSoup(raw_review).get_text()
    #remove non letters
    letters_only = re.sub("[^a-zA-Z]"," ", review_text)
    
    words = letters_only.lower().split()
    
    stops = set(stopwords.words("english"))
    
    meaningful_words = [w for w in words if not w in stops]
    
    return(" ".join(meaningful_words))
    

#clean_review = review_to_words(train["review"][0])

# get clean reviews for all the reviews

num_reviews = train["review"].size

clean_train_reviews = []

print("Cleaning and parsing the training set movie reviews")
for i in range(num_reviews):
    if((i+1)%1000==0):
        print("Review %d of %d\n"%(i+1,num_reviews))
    clean_train_reviews.append(review_to_words(train["review"][i]))




print("Creating the bag of words...\n")

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", \
                             tokenizer = None, \
                             preprocessor = None, \
                             stop_words = None,
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_train_reviews)
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

#f = open("voc.txt","w")
#for i in range(len(vocab)):
#    f.write(vocab[i])
#    f.write("\n")
#f.close()

import numpy as np

dist = np.sum(train_data_features,axis=0)

for tag, count in zip(vocab,dist):
    print(count,tag)


print("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data_features, train["sentiment"])

Y_train_predict = forest.predict(train_data_features)
from sklearn.metrics import accuracy_score
accuracy_score(train["sentiment"],Y_train_predict)

test = pd.read_csv("testData.tsv", header=0, delimiter = "\t", quoting=3)
print(test.shape)

num_reviews = len(test["review"])

clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...")

for i in range(num_reviews):
    if((i+1)%1000==0):
        print("Review %d of %d\n"%(i+1,num_reviews))
    clean_review = review_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

output = pd.DataFrame(data={"id":test["id"], "sentiment":result})

output.to_csv("Bag_of_words.csv",index=False,quoting=3)


