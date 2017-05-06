from __future__ import with_statement, print_function

import sys
#from optparse import make_option

from django.core.management.base import BaseCommand

from sitebias_core.models import Organization

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
        self.add_arguments(parser)
        return parser

    def handle(self, *args, **options):
        dryrun = options['dryrun']
        force = options['force']
        org_ids = list(map(int, [_ for _ in args if _.strip().isdigit()]))
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
