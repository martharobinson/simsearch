from transformers import CLIPProcessor, CLIPModel
import torch


def generate_clip_embeddings(model, processor, images=None, texts=None, device=None):
    """
    Generates CLIP embeddings for images and/or texts.
    Handles single input or batches.
    Returns a dict with 'image_embeds' and/or 'text_embeds'.
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    result = {}

    if images is not None:
        # Accept single image or list of images
        if not isinstance(images, list):
            images = [images]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)
        result["image_embeds"] = image_embeds.cpu()

    if texts is not None:
        # Accept single text or list of texts
        if not isinstance(texts, list):
            texts = [texts]
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
        result["text_embeds"] = text_embeds.cpu()

    return result


def load_clip_mps(model_name="openai/clip-vit-base-patch16"):
    """
    Loads CLIP model and processor from transformers, moves model to MPS for inference.
    Returns (model, processor) tuple.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
    return model, processor
