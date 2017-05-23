"""
http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/
"""
from __future__ import with_statement, print_function

import os
import string # pylint: disable=deprecated-module
import random
import traceback
#from math import e
import pickle

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
#from nltk.corpus import movie_reviews

#nltk.download('movie_reviews') # if necessary...
nltk.download('punkt') # if necessary...

def normalize(text):
    '''remove punctuation, lowercase, stem'''

    stemmer = nltk.stem.porter.PorterStemmer()
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

    def stem_tokens(tokens):
        lst = []
        for item in tokens:
            try:
                lst.append(stemmer.stem(item))
            except IndexError as exc:
                traceback.print_exc()
        return lst

    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

def word_feats(words):
    return dict((word, True) for word in words)

NEG = 'neg'
POS = 'pos'

#negids = movie_reviews.fileids(NEG)
#posids = movie_reviews.fileids(POS)

#negfeats = [(word_feats(movie_reviews.words(fileids=[f])), NEG) for f in negids]
#posfeats = [(word_feats(movie_reviews.words(fileids=[f])), POS) for f in posids]

#negcutoff = int(len(negfeats)*3/4)
#poscutoff = int(len(posfeats)*3/4)
#print('cutoffs:', negcutoff, poscutoff)

#trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
#testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
#print('train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats)))

#classifier = NaiveBayesClassifier.train(trainfeats)
#print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))

CLASSIFIER_FN = 'sentiment_classifier.pkl'

def train_classifier(fn=CLASSIFIER_FN):
    if not os.path.isfile(fn):
        print('Training classifier...')
        samples = []

        #https://github.com/cjhutto/vaderSentiment#resources-and-dataset-descriptions
        #FORMAT: the file is tab delimited with ID, MEAN-SENTIMENT-RATING, and TEXT-SNIPPET
        with open('sitebias_core/fixtures/nytEditorialSnippets_GroundTruth.txt', 'r') as fin:
            for line in fin.readlines():
                if not line.strip():
                    continue
                _id, rating, text = line.split('\t')
                rating = float(rating)
                rating_cls = POS if rating >= 0 else NEG
                samples.append((word_feats(normalize(text)), rating_cls))

        random.shuffle(samples)
        cutoff = int(len(samples)*3/4)

        trainfeats = samples[:cutoff]
        testfeats = samples[cutoff:]

        _classifier = NaiveBayesClassifier.train(trainfeats)
        print('accuracy:', nltk.classify.util.accuracy(_classifier, testfeats))

        _classifier = NaiveBayesClassifier.train(samples)
        print('accuracy(all):', nltk.classify.util.accuracy(_classifier, testfeats))

        with open(fn, 'wb') as fout:
            pickle.dump(_classifier, fout)

    return pickle.load(open(fn, 'rb'))

classifier = train_classifier()

def sentiment(text):
    words = normalize(text)
    features = word_feats(words)
    #cls = classifier.classify(features)
    cls = classifier.prob_classify(features)
    #return cls.prob(POS)
    return cls.logprob(POS)
