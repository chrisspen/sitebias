#!/bin/bash
# Runs the full data update pipeline.
set -e
. .env/bin/activate
cd src/sitebias

echo "Refreshing feeds..."
./manage.py refreshfeeds --days=0 --processes=1 --traceback

echo "Retrieving article text..."
./manage.py extract_post_articles --traceback

#echo "Calculating n-grams..."
#./manage.py extract_post_ngrams --traceback

echo "Updating criteria..."
./manage.py update_organizations --traceback --features
./manage.py update_organizations --traceback --clusters --criterias=6
