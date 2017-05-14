from __future__ import with_statement, print_function

import sys
#from optparse import make_option

from django.conf import settings
from django.core.management.base import BaseCommand

from sitebias_core.models import Organization, OrganizationFeature, ClusterCriteria

class Command(BaseCommand):

    help = "Test weka"

    args = ''

    def create_parser(self, prog_name, subcommand):
        parser = super(Command, self).create_parser(prog_name, subcommand)
        self.add_arguments(parser)
        return parser

    def handle(self, *args, **options):
        
        pass
