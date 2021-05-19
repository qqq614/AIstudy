import re    
import json
import logging
import torch
import os
import argparse
import boto3
import sys
from transformers import BertForSequenceClassification
from tokenization_kobert import KoBertTokenizer
from scipy.special import softmax

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'model_dir : {model_dir}')
    print(f'file list : {os.listdir(model_dir)}')
    model_dir='./ko-KR'
    model=BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer=KoBertTokenizer.from_pretrained(model_dir)
    return (tokenizer, model.to(device))

def input_fn(request_body, request_content_type):
    """An input_fn that deserializes and prepares prediction input"""
    if request_content_type == "application/json;charset=UTF-8" or request_content_type == "application/json":
        logger.info(f'request_body : {request_body}')
        request=json.loads(request_body)
        if 'query' not in request:
            raise ValueError("query not present in request body")
        query=request['query']
        logger.info(f'received query: {query}')
        return query
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(query, tokenizer_model_pair):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """ predict_fn that performs prediction and give output"""
    tokenizer, model = tokenizer_model_pair
    encoding=tokenizer.encode_plus(query, padding='max_length', max_length=100, truncation=True, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        encoding_={key:t.to(device) for key,t in encoding.items()}
        outputs=model(**encoding_)
        logits=outputs[0]
        preds=logits.detach().cpu().numpy()
    preds=softmax(preds)
    return preds
