#!/bin/bash
set -e
if [ "$#" -ne 1 ]
then
    echo "Usage: restore_db.sh <file.sql.gz>"
    exit 1
fi
if [ ! -f "$1" ]
then
    echo "File $1 does not exist."
    exit 1
fi
./init_db.sh
gunzip < $1 | pg_restore -U postgres --format=c --create --dbname=sitebias
