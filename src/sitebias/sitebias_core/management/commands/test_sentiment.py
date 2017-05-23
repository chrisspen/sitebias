"""
http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/
"""
from __future__ import with_statement, print_function

import time
from datetime import date
from math import e
#from optparse import make_option

from django.core.management.base import BaseCommand

from sitebias_core.models import Organization, OrganizationFeature

from sitebias_core.sentiment import sentiment

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
