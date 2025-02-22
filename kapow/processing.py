import json
import time
import torch

from kapow.embedding import embed
from kapow.qwen_model import tokenizer, device, encoder_model


def serialize_function_call(function_name, args, kwargs, arg_names):
    arg_json = json.dumps({
        "function_name": function_name,
        "time": time.time_ns(),
        "args": args,
        "kwargs": kwargs,
        "arg_names": arg_names,
    })
    return arg_json

def get_embedding_from_text(text):
    return embed(f"JSON representation of: {text}")

def process_nn_output(nn_output):
    nn_output_json = json.dumps({
        "output_from_nn": nn_output.tolist()
    })
    return embed(f"JSON representation of NN output:\n\n{nn_output_json}")

def get_first_token_embedding(text):
    """
    Generate an embedding vector that would produce the first token of the given text.
    This is done by encoding the text and getting the embedding of the first token.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    # Get the embedding of the first token
    first_token_embedding = outputs.last_hidden_state[0, 0].tolist()
    return first_token_embedding

def initialize_signature(sig):
    """
    Recursively initialize a signature datastructure composed of types.
    For each type, call its constructor (e.g. int() returns 0, str() returns "").
    For lists, tuples, and dicts, process recursively.
    """
    if isinstance(sig, dict):
        return {k: initialize_signature(v) for k, v in sig.items()}
    elif isinstance(sig, list):
        return [initialize_signature(item) for item in sig]
    elif isinstance(sig, tuple):
        return tuple(initialize_signature(item) for item in sig)
    elif isinstance(sig, type):
        try:
            return sig()
        except Exception:
            return None
    else:
        return sig