import time

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
