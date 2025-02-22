from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device = torch.device("cuda")
# Load the local transformer model Qwen2.5-1.5B for decoding.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
model.to(device)
model.eval()
def decode(embedding_vector, max_length=200, verbose=True):
    # Convert the embedding vector to a tensor and reshape to (batch_size, seq_len, hidden_size).
    # Here we assume a single-token sequence.
    inputs_embeds = torch.tensor(embedding_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device)
    if verbose:
        print("Decoding diagnostics:")
        print("Input embedding vector length:", len(embedding_vector))
        print("Inputs_embeds shape:", inputs_embeds.shape)
    # Run generate with forced BOS token using beam search.
    generated_ids = model.generate(
        inputs_embeds=inputs_embeds,
        max_length=max_length,
        forced_bos_token_id=tokenizer.bos_token_id,
        do_sample=False,
        num_beams=5,
    )
    if verbose:
        print("Generated ids tensor shape:", generated_ids.shape)
        # Print only the first 10 token ids as a sample for diagnostic purposes.
        print("Generated ids sample:", generated_ids[0][:10])
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    if verbose:
        print("Decoded text:", text)
    return text