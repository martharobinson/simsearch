import torch
from torchviz import make_dot
from simsearch.clip_embeddings import load_lora_clip
import os

# Path to LoRA model directory
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/clip_vit_lora_lightning/final_lora')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../diagrams/load_model_architecture')

# Load LoRA model
lora_model, processor = load_lora_clip(MODEL_DIR)

# Create dummy input (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224).to(lora_model.device)

# Forward pass
output = lora_model(pixel_values=dummy_input)

# Visualize and save computation graph
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
dot = make_dot(output.pooler_output, params=dict(lora_model.named_parameters()))
dot.render(OUTPUT_PATH, format="png")

print(f"Model architecture graph saved to {OUTPUT_PATH}.png")
