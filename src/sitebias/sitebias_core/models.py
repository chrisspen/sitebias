from datetime import date
import sys
import calendar
from collections import defaultdict
from math import log

from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

from picklefield.fields import PickledObjectField

print('Loading scipy...')
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
#from scipy.sparse import vstack

from feedz.feedutil import search_links_url
from feedz.models import Feed, Post#, NGram

from .clustering.kmeanseven import KMeansEven

class BaseModel(models.Model):

    created = models.DateTimeField(auto_now_add=True)

    updated = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class OrganizationManager(models.Manager):

    def get_by_natural_key(self, homepage):
        return self.get(homepage=homepage)

class Organization(BaseModel):

    objects = OrganizationManager()

    name = models.CharField(max_length=100, blank=False, null=False)

    homepage = models.URLField(unique=True)

    rss_homepage = models.URLField(unique=True, null=True, blank=True)

    #feeds = models.ManyToManyField('feedz.Feed')

    feed_count = models.PositiveIntegerField(default=0, editable=False, verbose_name='total feed count')

    active_feed_count = models.PositiveIntegerField(default=0, editable=False)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.name == other.name

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.name < other.name

    def natural_key(self):
        return (self.homepage,)

    def check_homepage_for_feeds(self, dryrun=False):
        homepages = [self.homepage, self.rss_homepage]
        for homepage in homepages:
            if not homepage:
                continue
            links = search_links_url(url=homepage, depth=3)
            for link in links:
                print('link:', link)
                if not dryrun:
                    feed = OrganizationFeed.objects.get_or_create(
                        organization=self,
                        feed_url=link,
                        defaults=dict(is_active=False))[0]
        self.save()

    def save(self, *args, **kwargs):
        if self.id:
            self.feed_count = self.feeds.all().count()
            self.active_feed_count = self.feeds.filter(is_active=True).count()
        super(Organization, self).save(*args, **kwargs)

class OrganizationFeed(Feed):

    organization = models.ForeignKey(Organization, related_name='feeds')

class OrganizationFeatureManager(models.Manager):

    def get_stale(self):
        return self.filter(fresh=False)

class OrganizationFeature(BaseModel):

    objects = OrganizationFeatureManager()

    organization = models.ForeignKey(Organization, related_name='features')

    start_date = models.DateField(blank=False, null=False, editable=False)

    end_date = models.DateField(blank=False, null=False, editable=False)

    fresh = models.BooleanField(default=False, editable=False)

    ngram_counts = PickledObjectField(
        blank=True,
        null=True,
        compress=True,
        help_text='{ngram:count}')

    class Meta:
        unique_together = (
            ('organization', 'start_date', 'end_date'),
        )
        ordering = (
            'organization',
            'start_date',
        )
        
    def get_all_text(self, keywords=''):
        keywords = (keywords or '').strip().lower()
        posts = Post.objects.filter(
            feed__organizationfeed__organization=self.organization,
            date_published__gte=self.start_date,
            date_published__lte=self.end_date,
            article_content_length__gt=0)
        for post in posts:
            text = (post.article_content or '').strip().lower()
            if keywords and keywords not in text:
                continue
            yield text

    @classmethod
    def update_all(cls, force=False):

        # Find all unique (org, date_published) combinations.
        qs = Post.objects.all().values('date_published', 'feed__organizationfeed__organization').distinct()
        total = qs.count()
        i = 0
        for r in qs:
            i += 1
            sys.stdout.write('\rCreating org feature stub %i of %i...' % (i, total))
            sys.stdout.flush()
            start_date, end_date = dt_to_month_range(r['date_published'])
            cls.objects.get_or_create(
                organization_id=r['feed__organizationfeed__organization'],
                start_date=start_date,
                end_date=end_date,
                defaults=dict(fresh=False))

        if force:
            qs = cls.objects.all()
        else:
            qs = cls.objects.get_stale()
        total = qs.count()
        i = 0
        for r in qs:
            i += 1
            ngrams = defaultdict(int)
            posts = Post.objects.filter(
                feed__organizationfeed__organization=r.organization,
                date_published__gte=r.start_date,
                date_published__lte=r.end_date,
                article_ngrams_extracted=True)
            total_posts = posts.count()
            j = 0
            for post in posts:
                j += 1
                if i == 1 or j == 1 or i == total or j == total_posts or not j % 10:
                    sys.stdout.write('\rUpdating org feature %i of %i (post %i of %i)...' % (i, total, j, total_posts))
                    sys.stdout.flush()
                for gram, cnt in post.article_ngram_counts.items():
                    ngrams[gram] += cnt
            r.ngram_counts = dict(ngrams)
            r.fresh = True
            r.save()

def dt_to_month_range(dt):
    try:
        _, total_days = calendar.monthrange(dt.year, dt.month)
        start_date = date(dt.year, dt.month, 1)
        end_date = date(dt.year, dt.month, total_days)
        return start_date, end_date
    except ValueError:
        print('Unable to convert to range, invalid date:', dt)
        raise

# Create and update an OrganizationFeature after every feedz.Post is saved.
@receiver(post_save, sender=Post)
def post_post_save(sender, **kwargs):
    post = kwargs.pop('instance')
    orgfeed = OrganizationFeed.objects.get(feed_ptr=post.feed)
    organization = post.feed
    start_date, end_date = dt_to_month_range(post.date_published)
    feature, _ = OrganizationFeature.objects.get_or_create(
        organization=orgfeed.organization,
        start_date=start_date,
        end_date=end_date)
    feature.fresh = False
    feature.save()

class TFIDF(object):

    def __init__(self):
        # https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Definition
        # As a term appears in more documents, the ratio inside the logarithm approaches 1, bringing the idf and tf-idf closer to 0.
        # Thus a higher tf/idf weight -> the more important the term.
        self._tf = defaultdict(lambda: defaultdict(float)) # {d:{t: count}}

        # Total term count in all documents.
        self._tf_counts = defaultdict(float)

        # Total document terms in all documents the term appeared in.
        self._tf_sum = defaultdict(float)

        # Total number of documents.
        self._idf_n_set = set()

        # Count of documents that contain term.
        self._idf_t_set = defaultdict(set) # {t: set(d containing t)}

    def update(self, doc, term, count, all_count=0):
        self._tf[doc][term] += count
        self._idf_n_set.add(doc)
        self._idf_t_set[term].add(doc)
        self._tf_counts[term] += count
        self._tf_sum[term] += all_count

    def get_tf(self, term, doc=None):
        """
        In the case of the term frequency tf(t,d), the simplest choice is to use the raw count of a term in a document,
        i.e. the number of times that term t occurs in document d.
        If we denote the raw count by ft,d, then the simplest tf scheme is tf(t,d) = ft,d.

        term frequency adjusted for document length : ft,d / (number of words in d)
        """
        if doc:
            return self._tf[doc][term]
        else:
            return self._tf_counts[term]/self._tf_sum[term]

    def get_idf(self, term, D=None):
        """
        It is the logarithmically scaled inverse fraction of the documents that contain the word,
        obtained by dividing the total number of documents by the number of documents containing the term,
        and then taking the logarithm of that quotient.
        """
        D = D or self._idf_n_set
        N = len(self._idf_n_set)
        num_docs_with_term = len(self._idf_t_set[term]) + 1
        v = log(float(N)/num_docs_with_term)
        return v

    def get(self, term, doc=None, D=None):
        D = D or self._idf_n_set
        tf = self.get_tf(term=term, doc=doc)
        idf = self.get_idf(term=term, D=D)
        return tf * idf

class ClusterCriteria(BaseModel):
    
    KMEANS = 'kmeans'
    KMEANS_EVEN = 'kmeans-even'
    ALGORITHM_CHOICES = [
        (KMEANS, 'K-Means'),
        (KMEANS_EVEN, 'K-Means-Even'),
    ]

    number_of_clusters = models.PositiveIntegerField(default=2, blank=False, null=False)

    algorithm = models.CharField(max_length=50, choices=ALGORITHM_CHOICES, default=KMEANS, blank=False, null=False)

    start_date = models.DateField(blank=False, null=False)

    class Meta:
        unique_together = (
            ('number_of_clusters', 'algorithm'),
        )

    def __str__(self):
        return u'Clusters=%i, Algo=%s, Start=%s' % (self.number_of_clusters, self.algorithm, self.start_date)

    def save(self, *args, **kwargs):

        self.start_date = date(self.start_date.year, self.start_date.month, 1)

        super(ClusterCriteria, self).save(*args, **kwargs)

    @classmethod
    def update_all(cls, criterias_ids=None):

        criterias = cls.objects.all()
        if criterias_ids:
            criterias = criterias.filter(id__in=criterias_ids)
        for criteria in criterias:

            # Create label stubs.
            qs = OrganizationFeature.objects.filter(start_date__gte=criteria.start_date)
            total = qs.count()
            i = 0
            for r in qs:
                i += 1
                sys.stdout.write('\rCreating cluster label for feature %i of %i...' % (i, total))
                sys.stdout.flush()
                ClusterLabel.objects.get_or_create(
                    criteria=criteria,
                    organization=r.organization,
                    start_date=r.start_date,
                    end_date=r.end_date)

            if criteria.algorithm == KMEANS:

                # Collect data.
                k_means_keys = [] # [orgfeature_id]
                k_means_data = [] # [{word: freq}]
                qs = ClusterLabel.objects.filter(criteria=criteria, index__isnull=True).values('organization', 'start_date', 'end_date').distinct()
                total_u = qs.count()
                u = 0
                for unq in qs:
                    u += 1
                    print('Processing combo %i of %i...' % (u, total))
                    features = OrganizationFeature.objects.filter(
                        organization=unq['organization'],
                        start_date=unq['start_date'],
                        end_date=unq['end_date'])
                    total_f = features.count()
                    f = 0
                    for feature in features:
                        f += 1
                        ngrams = feature.ngram_counts
                        count_sum = float(sum(ngrams.values()))
                        #print('Looking up ngrams for %s (%i of %i).' % (feature, f, total_f))
                        total_j = len(ngrams)
                        j = 0
                        ngram_freqs = {}
                        for k, v in ngrams.items():
                            #j += 1
                            #if j == 1 or j == total_j or not j % 100:
                                #sys.stdout.write('\rLooking up ngram %i of %i (%.2f%%)...' % (j, total_j, j/total_j*100))
                                #sys.stdout.flush()
                            #ngram_freqs[NGram.lookup(k).id] = v/count_sum
                            ngram_freqs[k] = v/count_sum
                        print('Converting ngram frequencies to sparse matrix.')
                        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html
                        #print('X:', X)
                        k_means_keys.append(feature.id)
                        k_means_data.append(ngram_freqs)

                # Cluster and apply labels.
                print('Clustering...')
                v = DictVectorizer(sparse=True, dtype=float)
                X = v.fit_transform(k_means_data)
                kmeans = KMeans(n_clusters=criteria.number_of_clusters, random_state=0).fit(X)
                for label, feature_id in zip(kmeans.labels_, k_means_keys):
                    print('Updating feature %s with label %s.' % (feature_id, label))
                    ofeature = OrganizationFeature.objects.get(id=feature_id)#.update(index=label)
                    clabel = ClusterLabel.objects.get_or_create(
                        criteria=criteria,
                        organization=ofeature.organization,
                        start_date=ofeature.start_date,
                        end_date=ofeature.end_date,
                    )[0]
                    clabel.index = label
                    clabel.save()

            elif criteria.algorithm == KMEANS_EVEN:

                print('Collecting data for K-means-even...')
                tfidf = TFIDF()
                all_terms = set()
                term_weights = {}
                k_means_keys = [] # [orgfeature_id]
                data_id_to_feature_id = {} # {id: orgfeature_id}
                k_means_data = [] # [{word: freq}]
                qs = ClusterLabel.objects.filter(
                    criteria=criteria,
                    #index__isnull=True
                )
                qs.update(index=None)
                qs = qs.values('organization', 'start_date', 'end_date').distinct()
                total_u = qs.count()
                u = 0
                for unq in qs:
                    u += 1
                    print('Processing combo %i of %i...' % (u, total))
                    doc = (unq['organization'], unq['start_date'], unq['end_date'])
                    features = OrganizationFeature.objects.filter(
                        organization=unq['organization'],
                        start_date=unq['start_date'],
                        end_date=unq['end_date'])
                    total_f = features.count()
                    f = 0
                    for feature in features:
                        f += 1
                        ngrams = feature.ngram_counts
                        count_sum = float(sum(ngrams.values()))
                        #print('Looking up ngrams for %s (%i of %i).' % (feature, f, total_f))
                        total_j = len(ngrams)
                        j = 0
                        ngram_freqs = {}
                        for k, v in ngrams.items():

                            # Exclude 1,2 grams
                            #space_cnt = k.count(' ')
                            #if space_cnt <= 1:
                                #continue
                            ## Exclude 5 grams.
                            #if space_cnt >= 4:
                                #continue

                            ngram_freqs[k] = v/count_sum
                            all_terms.add(k)
                            tfidf.update(doc=doc, term=k, count=v, all_count=count_sum)
                        k_means_keys.append(feature.id)
                        assert ngram_freqs, 'No feature data.'
                        data_id_to_feature_id[id(ngram_freqs)] = feature.id
                        k_means_data.append(ngram_freqs)

                print('Weighting all %i terms...' % len(all_terms))
                max_terms = 500000
                for term in all_terms:
                    term_weights[term] = tfidf.get(term=term)
                term_weights = list(term_weights.items())
                term_weights.sort(key=lambda o: o[1], reverse=True)
                print('highest:', term_weights[0])
                print('lowest:', term_weights[-1])
                term_weights = term_weights[:max_terms]
                all_terms = set(term_weights)

                print('Deleting all inferior features...')
                total = len(k_means_data)
                i = 0
                min_features = 10000
                for data in k_means_data:
                    i += 1
                    keys_to_remove = set(data.keys()).difference(all_terms)
                    total_j = len(keys_to_remove)
                    j = 0
                    for k in keys_to_remove:
                        j += 1
                        if i == 1 or j == 1 or i == total or j == total_j or not j % 100000:
                            sys.stdout.write('\rProcessing row %i of %i, key %i of %i...' % (i, total, j, total_j))
                            sys.stdout.flush()
                        if len(data) > min_features:
                            data.pop(k, None)
                        else:
                            break
                    assert data, 'All features removed!'

                #cpriors = criteria.priors.all()
                priors = [] # [(point, index)]
                for point in k_means_data:
                    feature_id = data_id_to_feature_id[id(point)]
                    ofeature = OrganizationFeature.objects.get(id=feature_id)
                    cpriors = criteria.priors.filter(organization=ofeature.organization)
                    if cpriors.exists():
                        priors.append((point, cpriors[0].index))

                print('Clustering using K-means-even...')
                print('point ids0:', len(data_id_to_feature_id))
                kmeans = KMeansEven(criteria.number_of_clusters).fit(k_means_data, priors=priors)
                point_ids = set()
                for label, cluster in enumerate(kmeans.clusters):
                    print('Labelling %i points in cluster %i...' % (len(cluster.points), label))
                    for point in cluster.points:
                        point_ids.add(id(point))
                        print('point_ids1:', len(point_ids))
                        #i = k_means_data.index(point)
                        #feature_id = k_means_keys[i]
                        feature_id = data_id_to_feature_id[id(point)]
                        print('Updating feature %s with label %s.' % (feature_id, label))
                        ofeature = OrganizationFeature.objects.get(id=feature_id)
                        clabel = ClusterLabel.objects.get_or_create(
                            criteria=criteria,
                            organization=ofeature.organization,
                            start_date=ofeature.start_date,
                            end_date=ofeature.end_date,
                        )[0]
                        clabel.index = label
                        clabel.save()

            else:
                raise NotImplementedError

class ClusterPrior(BaseModel):

    criteria = models.ForeignKey(ClusterCriteria, related_name='priors')

    organization = models.ForeignKey(Organization, related_name='priors')

    index = models.PositiveIntegerField(blank=False, null=False)

    class Meta:
        unique_together = (
            ('criteria', 'index'),
        )

    def save(self, *args, **kwargs):

        self.index = max(min(self.index, self.criteria.number_of_clusters - 1), 0)

        super(ClusterPrior, self).save(*args, **kwargs)

class ClusterLabel(BaseModel):

    criteria = models.ForeignKey(ClusterCriteria, related_name='labels')

    organization = models.ForeignKey(Organization, related_name='labels')

    start_date = models.DateField(blank=False, null=False)

    end_date = models.DateField(blank=False, null=False)

    index = models.PositiveIntegerField(blank=True, null=True, db_index=True)

    class Meta:
        unique_together = (
            ('criteria', 'organization', 'start_date', 'end_date'),
        )

class ClusterLink(BaseModel):
    
    criteria = models.ForeignKey(ClusterCriteria, related_name='links')
    
    from_organization = models.ForeignKey(Organization, related_name='from_links')

    start_date = models.DateField(blank=False, null=False)

    end_date = models.DateField(blank=False, null=False)
    
    to_organization = models.ForeignKey(Organization, related_name='to_links')
    
    weight = models.FloatField(blank=True, null=True)

    class Meta:
        unique_together = (
            ('criteria', 'from_organization', 'start_date', 'end_date', 'to_organization'),
        )
