#!/bin/bash

set -e

rm -rf /tmp/build
mkdir -p /tmp/build

python3 -m pip install -r lean-requirements.txt -t /tmp/build
python3 -m pip install . -t /tmp/build

cd /tmp/build
zip -r /io/lambda.zip *
