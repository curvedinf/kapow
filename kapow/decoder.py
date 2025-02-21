from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the local transformer model Qwen2.5-1.5B for decoding.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
model = AutoModelForSeq2SeqLM.from_pretrained("Qwen/Qwen2.5-1.5B")
model.eval()

def decode(embedding_vector, max_length=50):
    # Convert the embedding vector to a tensor and reshape to match expected encoder output dimensions.
    # Expected shape: (batch_size, seq_len, hidden_size). Here we assume a single token sequence.
    encoder_hidden_states = torch.tensor(embedding_vector, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    # Create an attention mask for the encoder output.
    encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.long)
    # Prepare a dummy decoder input using the beginning-of-sequence token.
    decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]])
    # Wrap the encoder outputs in an object with a last_hidden_state attribute.
    encoder_outputs = type("EncoderOutput", (object,), {"last_hidden_state": encoder_hidden_states})
    # Generate output text using the model's generate method.
    generated_ids = model.generate(
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=encoder_outputs,
        encoder_attention_mask=encoder_attention_mask,
        max_length=max_length
    )
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return text