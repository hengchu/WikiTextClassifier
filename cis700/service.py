import json
import onnx
from cis700.tokenizer import build_tokenizer

tok = build_tokenizer()

def lambda_handler(event, context):
    text = event.get('input_text')
    tokens = tok.tokenize(text)
    ids = tok.convert_tokens_to_ids(tokens)

    body = {
        'tokens': tokens,
        'ids': ids
    }

    return {
        'statusCode': 200,
        'body': body
    }
