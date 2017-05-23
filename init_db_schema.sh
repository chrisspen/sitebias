#!/bin/bash
set -e
if [ ! -f .env/bin/python ]
then
    echo "Run ./init_virtualenv.sh first."
    exit 1
fi
echo "Populating schema..."
.env/bin/python manage.py migrate --run-syncdb
.env/bin/python manage.py createsuperuser --username=admin
