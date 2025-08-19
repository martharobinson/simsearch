import torch
from torchviz import make_dot
from torchinfo import summary
from simsearch.clip_embeddings import load_lora_clip
import os

# Path to LoRA model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/clip_vit_lora_lightning/final_lora')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../diagrams/lora_model_architecture')

# Load LoRA model
lora_model, processor = load_lora_clip(MODEL_DIR)

# Create dummy input (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224).to(lora_model.device)

# Forward pass
output = lora_model(pixel_values=dummy_input)


# Print model summary using torchinfo
print("Model architecture summary:")
print(summary(lora_model, input_data={'pixel_values': dummy_input}, depth=3, col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], verbose=1))

# Print all LoRA parameters and their trainable status
print("\nLoRA parameters and trainable status:")
for name, param in lora_model.named_parameters():
	if "lora" in name.lower():
		print(f"{name}: shape={tuple(param.shape)}, trainable={param.requires_grad}")
