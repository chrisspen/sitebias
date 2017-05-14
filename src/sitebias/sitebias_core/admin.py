from django.contrib import admin

from feedz.models import Feed
from feedz.models import Feed

from admin_steroids.utils import view_link

from . import models # pylint: disable=no-name-in-module

class FeedInline(admin.TabularInline):

    model = models.OrganizationFeed

    fields = [
        'feed_url',
        'is_active',
        'feed_url_link',
    ]
#
    readonly_fields = [
        'feed_url_link',
    ]

    extra = 0

    def feed_url_link(self, obj):
        try:
            print(type(obj), obj, obj.feed_url)
            feed = Feed.objects.get(id=obj.id)
            return view_link(feed)
        except Exception as e:
            return str(e)
    feed_url_link.short_description = 'Edit'
    feed_url_link.allow_tags = True


class OrganizationAdmin(admin.ModelAdmin):

    list_display = [
        'name',
        'homepage',
        'feed_count',
        'active_feed_count',
        'post_count',
    ]

    readonly_fields = [
        'created',
        'updated',
        'feed_count',
        'active_feed_count',
        'post_count',
    ]

    search_fields = [
        'name',
        'homepage',
    ]

    fields = [
        'name',
        'homepage',
        'rss_homepage',
        'feed_count',
        'active_feed_count',
    ]

    inlines = [
        FeedInline,
    ]
    
    def post_count(self, obj):
        try:
            from feedz.models import Post
            posts = Post.objects.filter(feed__organizationfeed__organization=obj)
            return posts.count()
        except Exception as e:
            return str(e)
    post_count.short_description = 'posts'

class OrganizationFeatureAdmin(admin.ModelAdmin):

    list_display = [
        'organization',
        'start_date',
        'end_date',
        'fresh',
    ]

    list_filter = [
        'fresh',
    ]

    search_fields = [
        'organization__name',
    ]

    readonly_fields = [
        'organization',
        'start_date',
        'end_date',
        'fresh',
        'ngram_counts',
    ]

class ClusterPriorInline(admin.TabularInline):
    
    model = models.ClusterPrior
    
    fields = (
        'organization',
        'index',
    )
    
    raw_id_fields = [
        'organization',
    ]
    
    extra = 0

class ClusterCriteriaAdmin(admin.ModelAdmin):

    list_display = [
        'algorithm',
        'start_date',
        'number_of_clusters',
    ]

    readonly_fields = [
        'label_link',
    ]
    
    inlines = [
        ClusterPriorInline,
    ]

    def label_link(self, obj=None):
        try:
            if not obj or not obj.id:
                return ''
            qs = obj.labels.all()
            total = qs.count()
            return '<a class="btn button" href="../../../clusterlabel/?criteria__id__exact=%i">View %i</a>' % (obj.id, total)
        except Exception as e:
            return str(e)
    label_link.short_description = 'labels'
    label_link.allow_tags = True

class ClusterLabelAdmin(admin.ModelAdmin):

    list_display = [
        'criteria',
        'organization',
        'start_date',
        'end_date',
        'index',
    ]

    list_filter = [
        'criteria',
    ]

    readonly_fields = [
        'criteria',
        'organization',
        'start_date',
        'end_date',
        'index',
    ]

class ClusterLinkAdmin(admin.ModelAdmin):
    
    list_display = [
        'criteria',
        'from_organization',
        'start_date',
        'end_date',
        'to_organization',
        'weight',
    ]

    list_filter = [
        'criteria',
    ]

    readonly_fields = [
        'criteria',
        'from_organization',
        'start_date',
        'end_date',
        'to_organization',
        'weight',
    ]

admin.site.register(models.Organization, OrganizationAdmin)
admin.site.register(models.OrganizationFeature, OrganizationFeatureAdmin)
admin.site.register(models.ClusterCriteria, ClusterCriteriaAdmin)
admin.site.register(models.ClusterLabel, ClusterLabelAdmin)
admin.site.register(models.ClusterLink, ClusterLinkAdmin)
