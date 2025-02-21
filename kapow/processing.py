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
