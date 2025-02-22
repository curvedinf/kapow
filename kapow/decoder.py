import torch
from kapow.qwen_model import tokenizer, decoder_model, device
def decode(embedding_vector, max_new_tokens=200, verbose=True):
    # Diagnostic: print summary statistics of the embedding vector.
    if verbose:
        emb_min = min(embedding_vector)
        emb_max = max(embedding_vector)
        emb_mean = sum(embedding_vector) / len(embedding_vector)
        print("Embedding vector stats: min:", emb_min, "max:", emb_max, "mean:", emb_mean)
        print("Embedding vector length:", len(embedding_vector))
        print("Embedding vector sample (first 10):", embedding_vector[:10])
    # Convert the embedding vector to a tensor and reshape to (batch_size, seq_len, hidden_size).
    inputs_embeds = torch.tensor(embedding_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
    if verbose:
        print("Inputs_embeds shape:", inputs_embeds.shape)
    # Create an attention mask of ones that matches the first two dimensions of inputs_embeds.
    attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=device)
    if verbose:
        print("Attention mask shape:", attention_mask.shape)
    # Include the attention mask and set pad_token_id explicitly
    generated_ids = decoder_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        forced_bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=1,
        top_p=1.0,
    )
    if verbose:
        print("Generated ids tensor shape:", generated_ids.shape)
        print("Generated ids sample (first 10 tokens):", generated_ids[0][:10])
        print("Generated ids as list:", generated_ids[0].tolist())
    # Decode the generated tokens and ensure it starts with '['
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if not text.startswith('['):
        text = '[' + text
    if verbose:
        print("Decoded text:", text)
        tokenized_text = tokenizer.tokenize(text)
        print("Tokenized decoded text:", tokenized_text)
    return text