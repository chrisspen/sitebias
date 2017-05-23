from __future__ import with_statement, print_function

from django.core.management.base import BaseCommand

from sitebias_core.models import Organization

class Command(BaseCommand):

    help = "Show organization homepage URLs."

    args = ''

    def create_parser(self, prog_name, subcommand):
        parser = super(Command, self).create_parser(prog_name, subcommand)
        parser.add_argument('args', nargs="*")
        self.add_arguments(parser)
        return parser

    def handle(self, *args, **options):
        qs = Organization.objects.all()
        for org in qs:
            print(org.homepage)
