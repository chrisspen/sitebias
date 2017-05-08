#!/bin/bash
# Initializes the local Python virtual environment.
set -e
[ -d .env ] && rm -Rf .env
virtualenv -p python3 .env
. .env/bin/activate
pip install  --only-binary numpy,matplotlib,scipy,scikit-learn -r pip-requirements.txt
