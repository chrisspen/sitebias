# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-05-05 05:40
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('sitebias_core', '0002_auto_20170505_0522'),
    ]

    operations = [
        migrations.AddField(
            model_name='organization',
            name='rss_homepage',
            field=models.URLField(blank=True, null=True, unique=True),
        ),
    ]
