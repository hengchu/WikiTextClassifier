import boto3
import json
import numpy as np
import site
import pkg_resources
import zipfile
import os
import logging
import sys

from cis700.tokenizer import build_tokenizer

tok = build_tokenizer()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

PARAMS = {
    'vocab_size': 30522,
    'dim_embedding': 50,
    'num_heads': 5,
    'ff_num_features': 1024,
    'num_encoder_layers': 6,
    'max_seq_len': 256,
    'num_classes': 370,
}

COARSE_MAP_PATH = pkg_resources.resource_filename(
    'cis700', 'models/fine_categories.json')
COARSE_MAP = json.load(open(COARSE_MAP_PATH, 'r'))

TORCH_LIB_ZIP = {
    'Bucket': 'my-iex-data-dev',
    'Key': 'lambda/torch.zip',
    'Name': 'torch',
}

CAFFE2_LIB_ZIP = {
    'Bucket': 'my-iex-data-dev',
    'Key': 'lambda/caffe2.zip',
    'Name': 'caffe2',
}

EXTRA_PACKAGES_PATH = '/tmp/extra-pkgs'

if not os.path.exists(EXTRA_PACKAGES_PATH):
    os.makedirs(EXTRA_PACKAGES_PATH)

if EXTRA_PACKAGES_PATH not in sys.path:
    sys.path.append(EXTRA_PACKAGES_PATH)

def load_model():
    import torch
    from cis700.model import Classifier

    model = Classifier(**PARAMS)
    model_path = pkg_resources.resource_filename(
        'cis700', 'models/transformer-bootstrap2-s256-e50-h5-l6.epoch-9.step-265848.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def truncate(ids, max_seq_len):
  if len(ids) > max_seq_len:
    return ids[0:max_seq_len]
  return ids

def featurize_text(text, max_seq_len):
    tokens = tok.tokenize(text)
    ids = tok.convert_tokens_to_ids(tokens)
    ids = truncate(ids, max_seq_len)
    masks = [1] * len(ids)
    while len(ids) < max_seq_len:
        ids.append(0)
        masks.append(0)
    return ids, masks

def unzip_extra_package(file):
    with zipfile.ZipFile(file, 'r') as zf:
        zf.extractall(EXTRA_PACKAGES_PATH)

def install_lib(libdtor):
    if os.path.exists(EXTRA_PACKAGES_PATH + '/' + libdtor['Name']):
        logging.info('%s seems already installed, skipping...' % libdtor['Name'])
        return

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(libdtor['Bucket'])
    filename = '/tmp/' + libdtor['Key']
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'wb') as libpkg:
        bucket.download_fileobj(libdtor['Key'], libpkg)
    unzip_extra_package(filename)

def is_on_aws(context):
    if context and context.aws_request_id:
        return True
    return False

def lambda_handler(event, context):
    if is_on_aws(context):
        logger.info('installing torch...')
        install_lib(TORCH_LIB_ZIP)
        logger.info('installing caffe2...')
        install_lib(CAFFE2_LIB_ZIP)

        logger.info('finished package installation')

    import torch

    text = event.get('input_text')
    ids, masks = featurize_text(text, PARAMS['max_seq_len'])
    ids = torch.Tensor(ids).type(torch.LongTensor).unsqueeze(0)
    masks = torch.Tensor(masks).unsqueeze(0)

    model = load_model()
    with torch.no_grad():
        scores = model(ids, masks).squeeze().cpu().numpy()
        exp_scores = np.exp(scores)
        scores = (exp_scores / np.sum(exp_scores)).tolist()
        result = {}
        for i in range(len(scores)):
            category = COARSE_MAP[str(i)]
            result[category] = scores[i]

        return {
            'statusCode': 200,
            'body': result
        }
