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
    Generate an embedding vector that would produce the first token of the given assistant response.
    This is done by encoding the text and getting the embedding of the first important token of the assistant response.
    """
    # Apply chat template to get the full text
    text = tokenizer.apply_chat_template(
        target_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize the text and get the first token ID
    token_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=8000).input_ids[0]
    first_token_id = token_ids[0]
    
    # Get the embedding for the first token
    with torch.no_grad():
        embedding = encoder_model.get_input_embeddings()(torch.tensor([first_token_id]).to(device))
    
    # Return the embedding as a list
    return embedding[0].tolist()