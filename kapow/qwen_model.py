from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
device = torch.device("cuda")
# Load the tokenizer once
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Load the encoder model once for embedding
encoder_model = AutoModel.from_pretrained("Qwen/Qwen2.5-1.5B")
encoder_model.to(device)
encoder_model.eval()

# Load the decoder model once for generation
decoder_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
decoder_model.to(device)
decoder_model.eval()