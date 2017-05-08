# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-05-07 03:39
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion
import picklefield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('sitebias_core', '0003_organization_rss_homepage'),
    ]

    operations = [
        migrations.CreateModel(
            name='OrganizationFeature',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True)),
                ('start_date', models.DateField(editable=False)),
                ('end_date', models.DateField(editable=False)),
                ('fresh', models.BooleanField(default=False, editable=False)),
                ('ngram_counts', picklefield.fields.PickledObjectField(blank=True, editable=False, help_text='{ngram:count}', null=True)),
                ('organization', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='features', to='sitebias_core.Organization')),
            ],
        ),
        migrations.AlterUniqueTogether(
            name='organizationfeature',
            unique_together=set([('organization', 'start_date', 'end_date')]),
        ),
    ]
