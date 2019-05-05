FROM amazonlinux:2017.03

RUN yum -y install git \
    python36 \
    python36-devel \
    python36-pip \
    zip \
    gcc \
    && yum clean all

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install boto3

RUN rm -rf /tmp/cis700project

RUN mkdir -p /tmp/cis700project

WORKDIR /tmp/cis700project

COPY setup.py              ./
COPY lean-requirements.txt ./
COPY MANIFEST.in           ./
COPY cis700                cis700
