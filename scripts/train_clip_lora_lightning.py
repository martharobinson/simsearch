# Top-level dataset wrapper for image pairs
import os
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from simsearch.datasets import DeepFashionPairDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from PIL import Image

class CLIPLoRALightningModule(pl.LightningModule):
    def __init__(self, vision_model, lr=1e-4, weight_decay=1e-2):
        super().__init__()
        self.vision_model = vision_model
        self.lr = lr
        self.weight_decay = weight_decay

    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def info_nce_loss(self, emb_a, emb_b, temperature=0.07):
        # L2 normalize embeddings
        emb_a = self.l2_normalize(emb_a)
        emb_b = self.l2_normalize(emb_b)
        # Compute logits for both directions
        logits_ab = torch.matmul(emb_a, emb_b.T) / temperature
        logits_ba = torch.matmul(emb_b, emb_a.T) / temperature
        # Ensure labels are on the same device as logits
        device = logits_ab.device
        labels = torch.arange(len(emb_a), device=device)
        loss_ab = F.cross_entropy(logits_ab, labels)
        loss_ba = F.cross_entropy(logits_ba, labels)
        # Average the two losses
        loss = (loss_ab + loss_ba) / 2
        return loss

    def forward(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        # Main forward for PEFT/LoRA compatibility
        # Use argument parser logic to select correct input
        x = self._parse_input(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **kwargs)
        return self.vision_model(pixel_values=x).pooler_output

    def _parse_input(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
        # Argument parser for PEFT/LoRA compatibility
        if input_ids is not None:
            return input_ids
        elif pixel_values is not None:
            return pixel_values
        else:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor) and value.dim() == 4:
                    return value
            raise ValueError("No valid image tensor found in inputs.")

    def training_step(self, batch, batch_idx):
        start_time = time.time()
        img_a, img_b, _ = batch
        emb_a = self.forward(pixel_values=img_a)
        emb_b = self.forward(pixel_values=img_b)
        loss = self.info_nce_loss(emb_a, emb_b)
        batch_time = time.time() - start_time
        lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else self.lr
        self.log('train_loss', loss)
        self.log('learning_rate', lr, prog_bar=True)
        self.log('batch_time', batch_time)
        return loss

    def validation_step(self, batch, batch_idx):
        start_time = time.time()
        img_a, img_b, label = batch
        emb_a = self.forward(pixel_values=img_a)
        emb_b = self.forward(pixel_values=img_b)
        emb_a = self.l2_normalize(emb_a)
        emb_b = self.l2_normalize(emb_b)
        scores = F.cosine_similarity(emb_a, emb_b)
        batch_time = time.time() - start_time
        self.log('val_batch_time', batch_time)
        # Store outputs for epoch end
        if not hasattr(self, 'val_outputs'):
            self.val_outputs = []
        self.val_outputs.append({'scores': scores.cpu(), 'labels': label.cpu()})
        return {'scores': scores.cpu(), 'labels': label.cpu()}

    def on_validation_epoch_end(self):
        if hasattr(self, 'val_outputs') and self.val_outputs:
            all_scores = torch.cat([x['scores'] for x in self.val_outputs]).numpy()
            all_labels = torch.cat([x['labels'] for x in self.val_outputs]).numpy()
            if len(set(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_scores)
            else:
                auc = float('nan')
                self.print("[Warning] ROC AUC not computable: only one class present in validation labels.")
            self.log('val_roc_auc', auc)
            self.val_outputs = []  # Clear for next epoch
        else:
            self.log('val_roc_auc', float('nan'))

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.vision_model.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

class DeepFashionDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, eval_partition_path, batch_size=32, n_pairs=5000, seed=42, train_num_workers=0, val_num_workers=0):
        super().__init__()
        self.root_dir = root_dir
        self.eval_partition_path = eval_partition_path
        self.batch_size = batch_size
        self.n_pairs = n_pairs
        self.seed = seed
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


    def preprocess(self, img):
        # Ensure input is PIL.Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = self.augment(img)
        inputs = self.clip_processor(images=img, return_tensors="pt")
        # Only return pixel_values, never input_ids
        pixel_values = inputs.get('pixel_values', None)
        if pixel_values is None:
            raise ValueError("CLIPImageProcessor did not return 'pixel_values'.")
        return pixel_values.squeeze(0)

    def setup(self, stage=None):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Error handling for file paths
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        if not os.path.exists(self.eval_partition_path):
            raise FileNotFoundError(f"Eval partition file not found: {self.eval_partition_path}")
        try:
            base_dataset = DeepFashionPairDataset(self.root_dir, self.eval_partition_path, split="train", n_pairs=self.n_pairs)
        except Exception as e:
            raise RuntimeError(f"Failed to load DeepFashionPairDataset: {e}")
        wrapped_dataset = WrappedPairDataset(base_dataset, self.preprocess)
        val_size = int(0.02 * len(base_dataset))
        val_indices = np.random.choice(len(base_dataset), val_size, replace=False)
        train_indices = [i for i in range(len(base_dataset)) if i not in val_indices]
        self.train_dataset = Subset(wrapped_dataset, train_indices)
        self.val_dataset = Subset(wrapped_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.train_num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.val_num_workers)


class WrappedPairDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that applies preprocessing to image pairs.
    """
    def __init__(self, base_dataset, preprocess):
        self.base = base_dataset
        self.preprocess = preprocess
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img_a, img_b, label = self.base[idx]
        img_a = self.preprocess(img_a)
        img_b = self.preprocess(img_b)
        # Ensure only image tensors are returned
        if isinstance(img_a, dict):
            img_a = img_a['pixel_values'].squeeze(0)
        if isinstance(img_b, dict):
            img_b = img_b['pixel_values'].squeeze(0)
        return img_a, img_b, label

    # ...existing code...

if __name__ == "__main__":
    # Device setup handled by Lightning
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    vision_model.eval()
    for param in vision_model.parameters():
        param.requires_grad = False
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        # task_type="FEATURE_EXTRACTION"
    )
    vision_model = get_peft_model(vision_model, lora_config)
    vision_model.train()

    # Check trainable parameters after LoRA
    print("Trainable parameters after LoRA:")
    for name, param in vision_model.named_parameters():
        if param.requires_grad:
            print(f"  {name}")

    model = CLIPLoRALightningModule(vision_model)
    root_dir = "data/In-shop Clothes Retrieval Benchmark"
    eval_partition_path = os.path.join(root_dir, "Eval", "list_eval_partition.txt")
    # You can set worker counts here, e.g. train_num_workers=os.cpu_count(), val_num_workers=max(1, os.cpu_count()//2)
    data_module = DeepFashionDataModule(root_dir, eval_partition_path, train_num_workers=0, val_num_workers=0)

    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="models/clip_vit_lora_lightning/tb_logs",
        name="clip_lora_lightning"
    )
    trainer = pl.Trainer(
        max_epochs=3,
        log_every_n_steps=10,
        default_root_dir="models/clip_vit_lora_lightning",
        logger=tb_logger
    )
    trainer.fit(model, datamodule=data_module)

    # Save full Lightning checkpoint for resuming training
    checkpoint_path = os.path.join("models/clip_vit_lora_lightning", "final.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"Training complete. Full Lightning checkpoint saved to {checkpoint_path}.")

    # Save just the LoRA adapters for inference
    lora_save_path = os.path.join("models/clip_vit_lora_lightning", "final_lora")
    model.vision_model.save_pretrained(lora_save_path)
    print(f"Final LoRA adapters saved to {lora_save_path}.")
