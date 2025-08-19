import os
from torch.utils.data import DataLoader
import pandas as pd
import click
from tqdm import tqdm
import torch
from simsearch.datasets import DeepFashionDataset
from simsearch.clip_embeddings import load_lora_clip, generate_lora_clip_embeddings

def custom_collate(batch):
    images = [item["image"] for item in batch]
    filenames = [item["filename"] for item in batch]
    return {"image": images, "filename": filenames}

@click.command()
@click.option(
    "--data-root",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to DeepFashion dataset root directory.",
)
@click.option(
    "--split",
    required=True,
    type=click.Choice(["train", "gallery", "query"]),
    help="Dataset split to process.",
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    type=int,
    help="Batch size for embedding generation.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(),
    help="Path to save output embeddings (.npy).",
)
@click.option(
    "--lora-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to LoRA adapter directory (e.g., models/clip_vit_lora_lightning/final_lora)",
)
def main(data_root, split, batch_size, output, lora_dir):
    """Generate CLIP+LoRA image embeddings for DeepFashion dataset and save to file."""
    os.makedirs(os.path.dirname(output), exist_ok=True)

    dataset = DeepFashionDataset(root_dir=data_root, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, processor = load_lora_clip(lora_dir, device=device)

    first_batch = True
    total_count = 0
    num_batches = len(dataloader)
    for batch in tqdm(dataloader, desc=f"Embedding {split} images", total=num_batches):
        images = batch["image"]
        filenames = batch["filename"]
        embeds = generate_lora_clip_embeddings(model, processor, images=images, device=device)
        image_embeds = embeds.cpu().numpy()
        df = pd.DataFrame(image_embeds, index=filenames)
        df.index.name = "filename"
        df.to_csv(output, mode="w" if first_batch else "a", header=first_batch)
        total_count += len(filenames)
        first_batch = False
    print(f"Appended {total_count} embeddings to {output} as CSV")


if __name__ == "__main__":
    main()
