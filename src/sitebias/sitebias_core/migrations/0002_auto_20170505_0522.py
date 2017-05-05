# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-05-05 05:22
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('feedz', '__first__'),
        ('sitebias_core', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='OrganizationFeed',
            fields=[
                ('feed_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='feedz.Feed')),
            ],
            bases=('feedz.feed',),
        ),
        migrations.AddField(
            model_name='organization',
            name='active_feed_count',
            field=models.PositiveIntegerField(default=0, editable=False),
        ),
        migrations.AddField(
            model_name='organization',
            name='feed_count',
            field=models.PositiveIntegerField(default=0, editable=False, verbose_name='total feed count'),
        ),
        migrations.AddField(
            model_name='organizationfeed',
            name='organization',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='feeds', to='sitebias_core.Organization'),
        ),
    ]
