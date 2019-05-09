import sys
import pkg_resources
from cis700.tokenization import BertTokenizer

def build_tokenizer():
    vocab_path = pkg_resources.resource_filename('cis700', 'vocab/bert-base-uncased-vocab.txt')
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer

def main():
    # this is just an example of how to use the tokenizer
    tok = build_tokenizer()
    line = sys.stdin.readline()

    tokens = tok.tokenize(line)
    ids = tok.convert_tokens_to_ids(tokens)
    converted_back = tok.convert_ids_to_tokens(ids)

    print('>>>>> line')
    print(line)

    print('>>>>> tokens')
    print(tokens)

    print('>>>>> ids')
    print(ids)

    print('>>>>> tokens back from ids')
    print(converted_back)
