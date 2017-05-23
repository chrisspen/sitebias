from __future__ import with_statement, print_function

import sys
#from optparse import make_option

from django.conf import settings
from django.core.management.base import BaseCommand

from sitebias_core.models import Organization, OrganizationFeature, ClusterCriteria

class Command(BaseCommand):

    help = "Updates organizations."

    args = 'org.id'

    def create_parser(self, prog_name, subcommand):
        parser = super(Command, self).create_parser(prog_name, subcommand)
        parser.add_argument('args', nargs="*")
        parser.add_argument('--dryrun', action="store_true", default=False,
                    help="If given, no database changes will be made.")
        parser.add_argument('--force', action="store_true", default=False,
                    help="If given, all will be updated.")
        parser.add_argument('--feeds', action="store_true", default=False,
                    help="If given, feed links won't be checked.")
        parser.add_argument('--features', action="store_true", default=False,
                    help="If given, features won't be checked.")
        parser.add_argument('--clusters', action="store_true", default=False,
                    help="If given, clusters won't be checked.")
        parser.add_argument('--criterias', default='',
                    help="The cluster criterias to check.")
        parser.add_argument('--do-ngrams', action='store_true', default=False,
                    help="If given, updates n-grams aggregates.")
        self.add_arguments(parser)
        return parser

    def handle(self, *args, **options):

        #from sklearn.feature_extraction import DictVectorizer
        #from sklearn.cluster import KMeans
        #from scipy.sparse import coo_matrix, vstack

        #mydata = [
            #{'word1': 2, 'word3': 6, 'word7': 4},
            #{'word11': 1, 'word7': 9, 'word3': 2},
            #{'word5': 7, 'word1': 3, 'word9': 8},
        #]

        #kmeans_data = []
        #for raw_data in mydata:
            #cnt_sum = float(sum(raw_data.values()))
            #freqs = dict((k, v/cnt_sum) for k, v in raw_data.items())
            #kmeans_data.append(freqs)

        #v = DictVectorizer(sparse=True, dtype=float)
        #X = v.fit_transform(kmeans_data)

        #kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        #print(kmeans.labels_)

        #return
        settings.DEBUG = False
        dryrun = options['dryrun']
        force = options['force']
        org_ids = list(map(int, [_ for _ in args if _.strip().isdigit()]))

        if options['feeds']:
            if force:
                qs = Organization.objects.all()
            else:
                qs = Organization.objects.filter(feed_count=0)
            if org_ids:
                qs = qs.filter(id__in=org_ids)
            total = qs.count()
            print('%i pending records found.' % total)
            i = 0
            for org in qs:
                i += 1
                sys.stdout.write('\rUpdated %s (%i of %i)...' % (org, i, total))
                sys.stdout.flush()
                org.check_homepage_for_feeds(dryrun=dryrun)

        if options['features']:
            OrganizationFeature.update_all(do_ngrams=options['do_ngrams'])

        if options['clusters']:
            criterias = [int(_) for _ in options['criterias'].split() if _.strip().isdigit()]
            ClusterCriteria.update_all(criterias_ids=criterias)
