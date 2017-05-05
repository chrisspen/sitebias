from django.contrib import admin

from feedz.models import Feed

from admin_steroids.utils import view_link

from . import models

class FeedInline(admin.TabularInline):
      
    model = models.OrganizationFeed
     
    fields = [
#         'feed_url_link',
        'feed_url',
        'is_active',
    ]
#      
    readonly_fields = [
#         'feed_url_link',
    ]
    
    extra = 0
#      
#     def feed_url_link(self, obj):
#         return view_link(obj.feed)
#     feed_url_link.short_description = 'feed url'
#     feed_url_link.allow_tags = True
        

class OrganizationAdmin(admin.ModelAdmin):
    
    list_display = [
        'name',
        'homepage',
        'feed_count',
        'active_feed_count',
    ]
    
    readonly_fields = [
        'created',
        'updated',
        'feed_count',
        'active_feed_count',
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
    
admin.site.register(models.Organization, OrganizationAdmin)
