"""
http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/
"""
from __future__ import with_statement, print_function

import time
import sys
import string
import random
from datetime import date
import traceback
from math import e
#from optparse import make_option

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.encoding import force_text

from sitebias_core.models import Organization, OrganizationFeature, ClusterCriteria

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

nltk.download('movie_reviews') # if necessary...
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
            except IndexError as e:
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

def train_classifier():
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

    classifier = NaiveBayesClassifier.train(trainfeats)
    print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    
    classifier = NaiveBayesClassifier.train(samples)
    print('accuracy(all):', nltk.classify.util.accuracy(classifier, testfeats))
    return classifier

classifier = train_classifier()

def sentiment(text):    
    words = normalize(text)
    features = word_feats(words)
    #cls = classifier.classify(features)
    cls = classifier.prob_classify(features)
    #return cls.prob(POS)
    return cls.logprob(POS)

class Command(BaseCommand):

    help = "Test sentiment"

    args = ''

    def create_parser(self, prog_name, subcommand):
        parser = super(Command, self).create_parser(prog_name, subcommand)
        parser.add_argument('args', nargs="*")
        
        parser.add_argument('--keywords', default='',
                    help="Keywords to filter by.")
                    
        self.add_arguments(parser)
        return parser

    def handle(self, *org_id, **options):
        
        start_date = date(2017, 5, 1)
        #org_target = Organization.objects.get(id=int(org_id))
        #print('org_target:', org_target)
        #feature1 = OrganizationFeature.objects.get(organization=org_target, start_date=start_date)        
        #print('Getting text for target...')
        #text1 = ' '.join(feature1.get_all_text(keywords=options['keywords']))
        #print('len:', len(text1))

        scores = [] # [(sim to target, other org)]
        others = Organization.objects.all()
        total = others.count()
        i = 0
        for other_org in others:
        #for other_org in others[:2]:
            i += 1
            print('Processing org %i of %i...' % (i, total))
            
            feature2 = OrganizationFeature.objects.get(organization=other_org, start_date=start_date)
            
            print('Getting text for %s...' % other_org)
            text2 = ' '.join(feature2.get_all_text(keywords=options['keywords']))
            print('len:', len(text2))

            print('Calculating sentiment...')
            t0 = time.time()
            sim = sentiment(text2)
            td = time.time() - t0
            print('Calculated sim %s in %s seconds.' % (sim, td))
            scores.append((sim, other_org))
            
        print('-'*80)
        scores.sort()
        print('POS scores:')
        for lp, _org in scores:
            print('%.09f %s (%s)' % (lp, _org, e**lp))
        pass
