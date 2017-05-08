# -*- coding: utf-8 -*-
# Generated by Django 1.11 on 2017-05-07 05:15
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('sitebias_core', '0005_auto_20170507_0456'),
    ]

    operations = [
        migrations.AlterField(
            model_name='clusterlabel',
            name='criteria',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='labels', to='sitebias_core.ClusterCriteria'),
        ),
        migrations.AlterField(
            model_name='clusterlabel',
            name='index',
            field=models.PositiveIntegerField(blank=True, db_index=True, null=True),
        ),
        migrations.AlterUniqueTogether(
            name='clusterlabel',
            unique_together=set([('criteria', 'organization', 'start_date', 'end_date')]),
        ),
    ]
