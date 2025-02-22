import json
import time
from kapow.embedding import embed

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