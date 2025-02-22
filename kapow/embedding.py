import torch
from kapow.qwen_model import tokenizer, encoder_model, device

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8000)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = encoder_model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embedding_vector = outputs.pooler_output[0].tolist()
    else:
        last_hidden_state = outputs.last_hidden_state
        embedding_vector = torch.mean(last_hidden_state, dim=1)[0].tolist()
    return embedding_vector

def get_first_token_embedding(target_messages):
    """
    Generate an embedding vector that would produce the first token of the assistant response
    that contains a number. This is done by encoding the text and getting the embedding of
    the first token in the assistant response that contains a number.
    """
    # Apply chat template to get the full text
    text = tokenizer.apply_chat_template(
        target_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the text and get the token IDs
    token_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=8000).input_ids[0]
    
    # Find the start of the assistant response
    assistant_start = text.find(target_messages[-1]["content"])
    
    # Tokenize the assistant response separately
    assistant_text = text[assistant_start:]
    assistant_token_ids = tokenizer(assistant_text, return_tensors="pt", truncation=True, max_length=8000).input_ids[0]
    
    # Find the first token that contains a number
    for i, token_id in enumerate(assistant_token_ids):
        token = tokenizer.decode([token_id])
        if any(char.isdigit() for char in token):
            # Get the embedding for the first token containing a number
            with torch.no_grad():
                embedding = encoder_model.get_input_embeddings()(torch.tensor([token_id]).to(device))
            return embedding[0].tolist()
    
    # If no token contains a number, return the embedding of the first token
    with torch.no_grad():
        embedding = encoder_model.get_input_embeddings()(torch.tensor([assistant_token_ids[0]]).to(device))
    return embedding[0].tolist()