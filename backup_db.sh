#!/bin/bash
pg_dump -c -U sitebias --no-password --blobs --format=c --schema=public --host=localhost sitebias | gzip -c > db_postgresql_sitebias_localhost_$(date +%Y%m%d).sql.gz
