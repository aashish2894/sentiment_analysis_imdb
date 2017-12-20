#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 00:00:34 2017

@author: aashish
"""

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


    
sentences = []

print("Parsing the sentences from the training set")

for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing the sentences from the unlabeled set")

for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)


    
print(len(sentences))

print(sentences[0])
print(sentences[1])



import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

from gensim.models import word2vec
print("Training Model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
                          size=num_features, min_count=min_word_count, \
                          window = context, sample=downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save(model_name)



model.doesnt_match("france england germany berlin".split())
model.doesnt_match("paris berlin london austria".split())
model.most_similar("man")
model.most_similar("queen")
model.most_similar("awful")

