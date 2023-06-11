import runpod  # Required
import torch
from generate import generate_reply_HF
from memory import build_collection
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('TehVenom/Pygmalion-13b-Merged')
model = AutoModelForCausalLM.from_pretrained('TehVenom/Pygmalion-13b-Merged',torch_dtype=torch.bfloat16).cuda()

def handler(job):

    params = job["input"]
    history = params.pop('history')

    if params['build_collection']:
        collection = build_collection(history, params['character'])
        generate_reply_HF(history, tokenizer, model, collection, params, stopping_strings=['You:'])
        return collection
    else:
        embeddings
        reply = generate_reply_HF(history, tokenizer, model, embeddings, params, stopping_strings=['You:'])
        return reply

# This must be included for the serverless to work.
runpod.serverless.start({"handler": handler})  # Required