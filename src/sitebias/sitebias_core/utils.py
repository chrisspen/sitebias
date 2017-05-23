import calendar
from datetime import date

#from newspaper import fulltext
from newspaper import Article
from newspaper.article import ArticleException

def get_newspaper_text(url):
    try:
        #return fulltext(html)
        #t0 = time.time()
        a = Article(url)
        a.download()
        a.parse()
        #td = time.time() - t0
        #print('article download seconds: %s' % td)
        return a.text
    except ArticleException:
        return

def dt_to_month_range(dt):
    try:
        _, total_days = calendar.monthrange(dt.year, dt.month)
        start_date = date(dt.year, dt.month, 1)
        end_date = date(dt.year, dt.month, total_days)
        return start_date, end_date
    except ValueError:
        print('Unable to convert to range, invalid date:', dt)
        raise
