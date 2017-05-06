#from newspaper import fulltext
from newspaper import Article
from newspaper.article import ArticleException

def get_newspaper_text(url):
    try:
        #return fulltext(html)
        a = Article(url)
        a.download()
        a.parse()
        return a.text
    except ArticleException:
        return
