#!/bin/bash

set -e
set -o xtrace

rm -rf /tmp/build
mkdir -p /tmp/build

python3 -m pip install -r lean-requirements.txt -t /tmp/build
python3 -m pip install . -t /tmp/build

cd /tmp/build
zip -q -r /io/lambda.zip * \
    -x 'caffe2/*' \
    -x 'torch/*' \
    -x 'torch-*.dist-info/*'
zip -q -r /io/torch.zip torch/* torch-*.dist-info
zip -q -r /io/caffe2.zip caffe2/*
