# SimSearch: LoRA Approach for Efficient Model Fine-Tuning

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management and [DVC](https://dvc.org/) for data versioning.

1. **Install Poetry (if not already installed):**
	```bash
	curl -sSL https://install.python-poetry.org | python3 -
	```

2. **Install dependencies:**
	```bash
	poetry install
	```

3. **Download datset:**
Download the [The In-shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) dataset and add it to the ``data`` folder.

## Running with DVC

To reproduce experiments or run the pipeline:

1. **Run the DVC pipeline:**
	```bash
	poetry run dvc repro
	```
	This will execute all stages defined in `dvc.yaml` (e.g., embedding, training, evaluation) and ensure data and results are up to date.

2. **Check pipeline status:**
	```bash
	poetry run dvc status
	```

3. **View results:**
	Results and metrics will be saved in the `results/` directory. You can inspect evaluation outputs and compare model performance.

For more details on DVC usage, see the [DVC documentation](https://dvc.org/doc).

## Running the Backend and Frontend

This project provides two main applications:

### Backend API (`app.py`)

The backend serves the similarity search API. To run it:

```bash
poetry run python -m simsearch.app
```

This will start the backend server. By default, it loads the gallery embeddings and exposes endpoints for similarity search queries.

### Frontend UI (`ui_app.py`)

The frontend provides a user interface for querying the similarity search system. To run it:

```bash
poetry run python -m simsearch.ui_app
```

This will launch the UI, allowing you to upload query images and view retrieval results interactively.

**Note:** Ensure the backend is running before starting the frontend, as the UI communicates with the API server.


## Dataset

[The In-shop Clothes Retrieval Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) is a subset of the [DeepFashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html), designed to evaluate algorithms for large-scale fashion recognition and retrieval. The benchmark focuses on the challenging problem of identifying and retrieving images of clothing items from a gallery, given a query image. This task simulates real-world scenarios such as online shopping, where users may wish to find visually similar or identical items across different images, poses, and scales.

Key aspects of the benchmark include:
- **Cross-pose and cross-scale retrieval:** Images of the same clothing item are captured under varying poses, scales, and lighting conditions, testing the robustness of retrieval algorithms.
- **Rich annotations:** The dataset provides bounding boxes, fashion landmarks, item and attribute labels, segmentation masks, and dense pose annotations, enabling both supervised and unsupervised learning approaches.
- **Evaluation protocol:** The benchmark defines partitions for training, query, and gallery sets.

The goal is to advance research in computer vision for fashion, enabling more accurate and robust systems for clothes recognition, retrieval, and recommendation in practical applications.


## Vector Search Lookup

This project uses CLIP embeddings for fast image similarity search. Gallery image embeddings are precomputed and stored in a CSV file. At runtime, these embeddings are loaded and indexed using [hnswlib](https://github.com/nmslib/hnswlib), a high-performance library for approximate nearest neighbor search based on Hierarchical Navigable Small World (HNSW) graphs. See `simsearch/vector_search.py` for implementation details.

**How the lookup works:**

1. The query image is encoded into a CLIP embedding.
2. The embedding is normalized to a unit vector (for cosine similarity).
3. The system searches for the top-k most similar gallery images using cosine similarity via hnswlib's HNSW index.

![Project architecture](diagrams/Architecture.png)

### About HNSW

HNSW (Hierarchical Navigable Small World Graph) is a graph-based algorithm for approximate nearest neighbor search. It builds a multi-layered graph structure that allows fast and scalable search in high-dimensional spaces. HNSW is widely used for large-scale similarity search because it offers:

- **High recall and speed:** Finds nearest neighbors quickly with high accuracy.
- **Scalability:** Handles millions of vectors efficiently.
- **Cosine similarity support:** HNSW efficiently searches in spaces where vectors are normalized (unit length). In this project, CLIP embeddings are explicitly normalized in code before indexing and search, ensuring that cosine similarity is meaningful and robust for image retrieval.

**Why cosine similarity for CLIP embeddings?**

CLIP is trained with a contrastive loss that encourages similar images and texts to have embeddings pointing in similar directions, regardless of their magnitude. Cosine similarity measures the angle between vectors, making it robust to differences in scale and focusing on semantic similarity. Since CLIP embeddings often have varying norms, L2 distance can be dominated by magnitude rather than direction, which is less meaningful for semantic comparison. Cosine similarity is also less sensitive to outliers and works well for high-dimensional, dense representations like those produced by CLIP.

In this project, hnswlib's HNSW implementation is used to index and search CLIP embeddings for image similarity tasks. For details, see `simsearch/vector_search.py`.

## Evaluation Metrics for Similarity Search

To assess the performance of image retrieval, we use several standard metrics:

- **Recall@k:** Measures the fraction of queries for which at least one correct item is retrieved in the top-k results. High recall@k indicates that relevant items are frequently found among the top candidates.
- **Precision@k:** Measures the fraction of top-k retrieved items that are relevant. For this benchmark, relevance is binary (correct item ID or not).
- **Mean Average Precision (MAP):** Computes the average precision across all queries, considering the rank of each relevant item. MAP is robust to class imbalance and rewards methods that rank correct items higher.
- **Mean Reciprocal Rank (MRR):** Measures the average reciprocal rank of the first relevant item for each query. High MRR means correct items tend to appear near the top of the results.

These metrics are well-suited for the In-shop Clothes Retrieval Benchmark, where each query has a single correct gallery item (binary relevance).

## Evaluation Results

The following table summarizes the performance metrics for baseline and LORA models:

| Model   | Recall@5 | Precision@5 | MAP        | MRR        |
|---------|----------|-------------|------------|------------|
| Baseline| 0.7655   | 0.2743      | 0.6129     | 0.6396     |
| LORA    | 0.8949   | 0.4147      | 0.7715     | 0.8029     |

**Notes:**
- Higher values indicate better performance.
- Metrics are computed on the In-shop Clothes Retrieval Benchmark.

### About NDCG

**Normalized Discounted Cumulative Gain (NDCG)** is a popular metric for ranking tasks, especially when relevance is graded (e.g., 0, 1, 2, ...). In this benchmark, relevance is binary (0 or 1), so NDCG reduces to a form similar to MAP and MRR. While NDCG can still be computed, it does not provide additional insight beyond the chosen metrics when only binary relevance is available. If future tasks involve graded relevance (e.g., partial matches or attribute similarity), NDCG would become more valuable.

For most retrieval tasks in this project, recall@k, precision@k, MAP, and MRR are sufficient and interpretable.

## What is LoRA?

**LoRA (Low-Rank Adaptation)** is a technique for efficiently fine-tuning large neural networks. Instead of updating all weights, LoRA injects trainable low-rank matrices into specific layers (usually attention or linear layers), drastically reducing the number of parameters to train.

## Why Use LoRA?
- **Parameter Efficiency:** Only a small number of parameters are updated.
- **Faster Training:** Less memory and compute required.
- **Plug-and-Play:** Can be applied to pre-trained models without modifying their architecture.

## How LoRA Works


Instead of updating the full weight matrix `W` in a neural network layer, LoRA decomposes the update into two smaller matrices `A` and `B`:

Formula:
```text
W' = W + BA
```

Where:
- `W` is the original weight matrix (frozen)
- `A` and `B` are low-rank matrices (trainable)

**With rank 8, the LoRA update matrices have the following shapes:**

- Suppose the original weight matrix `W` has shape `(output_dim, input_dim)`.
- LoRA decomposes the update into two matrices:
	- `A` has shape `(8, input_dim)`
	- `B` has shape `(output_dim, 8)`

The product `BA` results in a matrix of shape `(output_dim, input_dim)`, matching `W`.

**Parameter count:**
Instead of training all `output_dim * input_dim` parameters, LoRA only trains `8 * (input_dim + output_dim)` parameters per layer, which is much smaller for typical transformer layers.

## Usage in This Project

This repository uses LoRA to fine-tune the Vision Transformer (ViT) part of CLIP for efficient similarity search in fashion images. Specifically, LoRA adapters are injected into the attention layers (`q_proj`, `v_proj`) of the ViT encoder, while the rest of the model remains frozen.

### How LoRA is used in CLIP's ViT Encoder

Below is a visual diagram showing LoRA applied to the attention blocks of the ViT encoder:  

![LoRA applied to ViT Attention](diagrams/LORA_Attention.png)

Only the LoRA adapters are updated during training; the rest of the ViT and CLIP model weights are kept frozen.

## Data Augmentation

To improve generalization and robustness, this project applies data augmentation to training images before they are processed by the CLIP model. The following augmentations are used:

- **Random Horizontal Flip:**  
	Images are randomly flipped left-to-right, helping the model learn invariance to orientation.
- **Color Jitter:**  
	Randomly adjusts brightness, contrast, saturation, and hue (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1). This encourages the model to focus on shape and texture rather than color, and makes it robust to lighting variations.

These augmentations are applied using `torchvision.transforms.Compose` and are performed on each image in the training set. Augmentation helps prevent overfitting and improves the model’s ability to generalize to new, unseen images.

## Loss Function

This project uses the **InfoNCE (Contrastive) Loss** to train the LoRA-adapted CLIP model. The loss encourages matching image pairs to have similar embeddings and non-matching pairs to be dissimilar.

- **How it works:**  
	For each batch, image embeddings are L2-normalized and compared using dot products (cosine similarity). The logits are divided by a temperature parameter to control the sharpness of the distribution.
- **Bidirectional loss:**  
	The loss is computed in both directions (A→B and B→A) and averaged, ensuring symmetry.
- **Implementation:**  
	The InfoNCE loss is implemented as cross-entropy over the similarity logits, with each image in the batch treated as the correct match for its counterpart.

This approach is effective for retrieval tasks, as it directly optimizes the model to produce discriminative and semantically meaningful embeddings.

See `scripts/train_clip_lora_lightning.py` for implementation details.

## References
- [LoRA: Low-Rank Adaptation of Large Language Models (arXiv)](https://arxiv.org/abs/2106.09685)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (arXiv)](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)