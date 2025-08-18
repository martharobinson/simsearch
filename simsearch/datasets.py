import os
from typing import Optional, Callable, Dict, Any
from PIL import Image
from torch.utils.data import Dataset


class DeepFashionDataset(Dataset):
    """
    PyTorch Dataset for DeepFashion In-shop Clothes Retrieval Benchmark.
    Loads images, item IDs, and attributes only.
    """

    HEADER_LINES = 2  # Number of header lines to skip in annotation files

    def __init__(
        self,
        root_dir: str,
        split: str = "train",  # "train", "query", "gallery"
        transform: Optional[Callable[[Image.Image], Any]] = None,
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.img_dir = os.path.join(root_dir, "Img")
        eval_path = os.path.join(root_dir, "Eval", "list_eval_partition.txt")
        self.image_names, self.items = self._load_eval_partition(eval_path, split)
        attr_items_path = os.path.join(
            root_dir, "Anno", "attributes", "list_attr_items.txt"
        )
        self.attr_dict = self._load_attr_annotations(attr_items_path)

    def _load_eval_partition(self, eval_path: str, split: str):
        image_names = []
        items = []
        try:
            with open(eval_path, "r") as f:
                lines = f.readlines()[self.HEADER_LINES :]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        continue
                    img_name, item_id, status = parts[:3]
                    if status == split:
                        image_names.append(img_name)
                        items.append(item_id)
        except Exception as e:
            print(f"Error loading eval partition: {e}")
        return image_names, items

    def _load_attr_annotations(self, attr_items_path: str):
        attr_dict = {}
        if not os.path.exists(attr_items_path):
            return attr_dict
        try:
            with open(attr_items_path, "r") as f:
                lines = f.readlines()[self.HEADER_LINES :]
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    item_id = parts[0]
                    attrs = parts[1:]
                    attr_dict[item_id] = attrs
        except Exception as e:
            print(f"Error loading attribute annotations: {e}")
        return attr_dict

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name = self.image_names[idx]
        item_id = self.items[idx]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = None
        if self.transform and img is not None:
            img = self.transform(img)
        sample = {
            "image": img,
            "filename": img_name,
            "item_id": item_id,
        }
        # Add attributes if available
        if item_id in self.attr_dict:
            sample["attributes"] = self.attr_dict[item_id]
        return sample
