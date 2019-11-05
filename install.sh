#!/bin/bash

## install package.

pip install -e .
pip install -r requirements.txt

## if packages are outdated, uncomment below
#pip list --outdated --format=freeze | grep -v '^\e' | cut -d = -f 1 | xargs -n1 pip install -r requirements.txt
