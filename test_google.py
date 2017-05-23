#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run like:

    export PYTHONIOENCODING=:backslashreplace; ./test_google.py
"""

import sys
import time
from datetime import date
import pprint
import traceback

import dateutil.parser

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

API_KEY = open('src/sitebias/sitebias/google_api_key.txt').read().strip()

#http://stackoverflow.com/questions/6562125/getting-a-cx-id-for-custom-search-google-api-python
#https://support.google.com/customsearch/answer/2649143?hl=en
#https://cse.google.com/cse/setup/basic
GOOGLE_ENGINE_ID = open('src/sitebias/sitebias/google_engine_id.txt').read().strip()

def get_pub_date(result):
    try:
        return dateutil.parser.parse(result['pagemap']['metatags'][0]['pubdate'])
        last_mod_date = dateutil.parser.parse(result['pagemap']['metatags'][0]['lastmod'])
    except KeyError as exc:
        try:
            return dateutil.parser.parse(result['pagemap']['metatags'][0]['date'])
        except KeyError as exc:
            try:
                return dateutil.parser.parse(result['pagemap']['metatags'][0]['lastmod'])
            except KeyError as exc:
                return

def get_last_mod_date(result):
    try:
        return dateutil.parser.parse(result['pagemap']['metatags'][0]['lastmod'])
    except KeyError as exc:
        return
                
def iter_results(term, site, start_date, end_date, max_results=100):
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    service = build("customsearch", "v1", developerKey=API_KEY)
    result_count = 0
    start_index = 1
    while 1:
        try:
            #https://developers.google.com/custom-search/json-api/v1/reference/cse/list
            res = service.cse().list(
                q=term,
                cx=GOOGLE_ENGINE_ID,
                #sort=date:r:20100101:20100201
                #sort='date:r:20100101:20110101',#0
                #sort='date:r:20150101:20160101',#800k
                #sort='date:r:20150101:20150201',
                #siteSearch='cnn.com',
                sort='date:r:%04i%02i%02i:%04i%02i%02i' % (
                    start_date.year, start_date.month, start_date.day,
                    end_date.year, end_date.month, end_date.day
                ),
                siteSearch=site,
                num=10,
                start=start_index,
            ).execute()
            #print('-'*80)
            results = res['items']
            total_results = res['searchInformation']['totalResults']
            if start_index == 1:
                print('total_results:', total_results)
            #print('%i results' % len(results))
            for i, result in enumerate(results):
                #print('-'*80)
                result_count += 1
                
                link = result['link']
                #print('i:', result_count)
                #print('link:', link)
                pub_date = get_pub_date(result)
                #print('pub_date:', pub_date)
                last_mod_date = get_last_mod_date(result) or pub_date
                #print('last_mod_date:', last_mod_date)
                if not pub_date:
                    print('Skipping result %i: no pub date' % (result_count,))
                    pprint.pprint(result)
                    continue
                
                if abs((last_mod_date - pub_date).days) > 30:
                    print('Skipping result %i: too much post-editing' % result_count)
                    continue
                
                yield dict(
                    link=link,
                    pub_date=pub_date,
                    last_mod_date=last_mod_date,
                )
                
                if result_count >= max_results:
                    return
            
            try:
                start_index = res['queries']['nextPage'][0]['startIndex']
                print('start_index:', start_index)
            except KeyError as exc:
                #print('No next page found: %s' % exc)
                #pprint.pprint(res)
                break

        except HttpError as exc:
            traceback.print_exc()
            break

def main():
    i = 0
    #site = 'cnn.com'
    site = 'npr.org'
    assert len(sys.argv) > 1
    for result in iter_results(term=' '.join(sys.argv[1:]), site=site, start_date=date(2015,1,1), end_date=date(2016,1,1)):
        i += 1
        print(i, result)
        time.sleep(1)

if __name__ == '__main__':
    main()
