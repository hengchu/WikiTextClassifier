import re
import subprocess
import torch

from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset
from cis700.tokenizer import build_tokenizer

def category_text_to_id(cat_map, cat_text, count):
  if cat_text in cat_map:
    return cat_map[cat_text], count
  else:
    cat_map[cat_text] = count
    return cat_map[cat_text], count+1

def count_lines(filepath):
  r = subprocess.Popen(['wc', '-l', filepath], stdout=subprocess.PIPE)
  r = r.communicate()
  output = r[0].decode('utf-8')
  output = output.strip(' ').split(' ')[0]
  return int(output)

# this needs to be global for multiprocessing purpose
_tokenizer   = None
_max_seq_len = None

class Feature:
  def __init__(self, text, ids, masks, fine_cat, coarse_cat, fine_cat_text, coarse_cat_text):
    self.text = text
    self.ids = ids
    self.masks = masks
    self.fine_cat = fine_cat
    self.coarse_cat = coarse_cat
    self.fine_cat_text = fine_cat_text
    self.coarse_cat_text = coarse_cat_text

  def __repr__(self):
    return self.__dict__.__repr__()

def _truncate(ids):
  if len(ids) > _max_seq_len:
    return ids[0:_max_seq_len]
  return ids

def _process(datum):
  tokens = _tokenizer.tokenize(datum[0])
  ids = _tokenizer.convert_tokens_to_ids(tokens)
  ids = _truncate(ids)
  masks = [1] * len(ids)
  while len(ids) < _max_seq_len:
    ids.append(0)
    masks.append(0)
  return Feature(datum[0], ids, masks, datum[1], datum[2], datum[3], datum[4])

def convert_to_features(data_list, max_seq_len):
  global _tokenizer
  global _max_seq_len

  _tokenizer = build_tokenizer()
  _max_seq_len = max_seq_len

  with Pool(processes=6) as pool:
    return list(pool.imap(_process, tqdm(data_list)))

  _tokenizer = None
  _max_seq_len = None

class DBPediaDataset(Dataset):
  def __init__(self, filepath, max_seq_len):
    self.fine_cat_map = {}
    fine_cat_count = 0
    self.coarse_cat_map = {}
    coarse_cat_count = 0
    self.data = []

    text_re = r'(".+"@en)'
    cat_re = r'\. (<[^<>]+>) (<[^<>]+>)$'
    tok = build_tokenizer()

    with open(filepath, 'r') as f:
      for line in tqdm(f, total=count_lines(filepath)):
        text_match = re.search(text_re, line)
        text = text_match.group(1).strip('"@en')
        cat_match = re.search(cat_re, line)
        fine_cat = cat_match.group(1)
        coarse_cat = cat_match.group(2)
        fine_cat_id, fine_cat_count = category_text_to_id(self.fine_cat_map, fine_cat, fine_cat_count)
        coarse_cat_id, coarse_cat_count = category_text_to_id(self.coarse_cat_map, coarse_cat, coarse_cat_count)
        self.data.append((text, fine_cat_id, coarse_cat_id, fine_cat, coarse_cat))

    self.reverse_fine_cat_map = {}
    self.reverse_coarse_cat_map = {}
    for k in self.fine_cat_map:
      v = self.fine_cat_map[k]
      assert v not in self.reverse_fine_cat_map
      self.reverse_fine_cat_map[v] = k
    for k in self.coarse_cat_map:
      v = self.coarse_cat_map[k]
      assert v not in self.reverse_coarse_cat_map
      self.reverse_coarse_cat_map[v] = k

    self.data = convert_to_features(self.data, max_seq_len)

  def fine_id2cat(self, id):
    return self.reverse_fine_cat_map[id]

  def coarse_id2cat(self, id):
    return self.reverse_coarse_cat_map[id]

  def num_fine_cats(self):
    return len(self.reverse_fine_cat_map)

  def num_coarse_cats(self):
    return len(self.reverse_coarse_cat_map)

  def __len__(self):
    return len(self.data)

  def get_feature(self, idx):
    return self.data[idx]

  def __getitem__(self, idx):
    dt = self.data[idx]
    return torch.Tensor(dt.ids), torch.Tensor(dt.masks), dt.fine_cat, dt.coarse_cat, idx
