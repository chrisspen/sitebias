#!/bin/bash
# Initializes the local PostgreSQL database.
# Requires that the local postgres user is either trusted or has a password in ~/.pgpass/
set -e
DB_NAME="${DB_NAME:-sitebias}"
DB_USERNAME="${DB_USERNAME:-sitebias}"
DB_PASSWORD="${DB_PASSWORD:-sitebias}"
echo "Dropping old database..."
psql --user=postgres --no-password --command="DROP DATABASE IF EXISTS $DB_NAME;"
psql --user=postgres --no-password --command="DROP ROLE IF EXISTS $DB_USERNAME;"
echo "Creating user..."
psql --user=postgres --no-password --command="CREATE USER $DB_USERNAME WITH PASSWORD '$DB_PASSWORD';"
echo "Creating database..."
psql --user=postgres --no-password --command="CREATE DATABASE $DB_NAME WITH OWNER=$DB_USERNAME ENCODING='UTF8' LC_CTYPE='en_US.UTF-8' LC_COLLATE='en_US.UTF-8'"
echo "Done!"
