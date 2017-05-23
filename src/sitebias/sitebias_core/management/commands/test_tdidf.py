"""
http://stackoverflow.com/questions/8897593/similarity-between-two-text-documents
"""
from __future__ import with_statement, print_function

import time
from datetime import date
import traceback
import string # pylint: disable=deprecated-module
#from optparse import make_option

from django.core.management.base import BaseCommand

from sitebias_core.models import Organization, OrganizationFeature

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

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

def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]

class Command(BaseCommand):

    help = "Test tf-idf"

    args = ''

    def create_parser(self, prog_name, subcommand):
        parser = super(Command, self).create_parser(prog_name, subcommand)
        parser.add_argument('args', nargs="*")

        parser.add_argument('--keywords', default='',
                    help="Keywords to filter by.")

        self.add_arguments(parser)
        return parser

    def handle(self, org_id, **options):

        start_date = date(2017, 5, 1)
        org_target = Organization.objects.get(id=int(org_id))
        print('org_target:', org_target)
        feature1 = OrganizationFeature.objects.get(organization=org_target, start_date=start_date)
        print('Getting text for target...')
        text1 = ' '.join(feature1.get_all_text(keywords=options['keywords']))
        print('len:', len(text1))

        scores = [] # [(sim to target, other org)]

        others = Organization.objects.all().exclude(id=org_target.id)
        total = others.count()
        i = 0
        for other_org in others:
            i += 1
            print('Processing org %i of %i...' % (i, total))

            feature2 = OrganizationFeature.objects.get(organization=other_org, start_date=start_date)
            assert feature1.id != feature2.id

            print('Getting text for %s...' % other_org)
            text2 = ' '.join(feature2.get_all_text(keywords=options['keywords']))
            print('len:', len(text2))
            assert text1 != text2
            assert len(text1) != len(text2)

            print('Calculating cosine similarity...')
            t0 = time.time()
            sim = cosine_sim(text1, text2)
            td = time.time() - t0
            print('Calculated sim %s in %s seconds.' % (sim, td))
            scores.append((sim, other_org))

        print('-'*80)
        scores.sort()
        for _sim, _org in scores:
            print('%.2f %s' % (_sim, _org))
