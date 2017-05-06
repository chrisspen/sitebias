from django.db import models

from feedz.feedutil import search_links_url
from feedz.models import Feed

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
