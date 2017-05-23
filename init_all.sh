#!/bin/bash
set -e
./init_virtualenv.sh
./init_db.sh
./init_db_schema.sh
