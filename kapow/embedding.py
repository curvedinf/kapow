import torch
from kapow.qwen_model import tokenizer, encoder_model, device

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embedding_vector = outputs.pooler_output[0].tolist()
    else:
        last_hidden_state = outputs.last_hidden_state
        embedding_vector = torch.mean(last_hidden_state, dim=1)[0].tolist()
    return embedding_vector