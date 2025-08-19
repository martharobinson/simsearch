from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
from pathlib import Path
import numpy as np
import uvicorn
from simsearch.clip_embeddings import load_clip_mps, generate_clip_embeddings
from simsearch.vector_search import load_embeddings, build_hnsw_index, search
from simsearch.clip_embeddings import load_lora_clip, generate_lora_clip_embeddings
import logging


app = FastAPI()

# Config
BASELINE_EMBEDDINGS_CSV = Path(__file__).parent.parent / "data/embeddings/baseline/gallery.csv"
LORA_EMBEDDINGS_CSV = Path(__file__).parent.parent / "data/embeddings/lora/gallery.csv"
K_DEFAULT = 5

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

logger.info("Loading baseline CLIP model and processor...")
baseline_model, processor = load_clip_mps()
logger.info("Loading baseline gallery embeddings...")
baseline_embeddings, baseline_filenames = load_embeddings(BASELINE_EMBEDDINGS_CSV)
logger.info(f"Loaded {len(baseline_embeddings)} baseline embeddings with shape {baseline_embeddings.shape}.")
logger.info(f'NaNs: {np.isnan(baseline_embeddings).sum()}, Infs: {np.isinf(baseline_embeddings).sum()}')
logger.info("Building baseline index...")
baseline_index = build_hnsw_index(baseline_embeddings)
logger.info("Loading LoRA model and processor...")
lora_model, lora_processor = load_lora_clip(
    model_dir=Path(__file__).parent.parent / "models/clip_vit_lora_lightning/final_lora",
)
logger.info("Loading LoRA gallery embeddings and building index...")
lora_embeddings, lora_filenames = load_embeddings(LORA_EMBEDDINGS_CSV)
lora_index = build_hnsw_index(lora_embeddings)


async def search_similar_images(
    file,
    k,
    model,
    processor,
    filenames,
    index,
    is_lora: bool = False
):
    logger.info(f"search_similar_images called (is_lora={is_lora}, k={k})")
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logger.info("Image loaded and converted to RGB")
    if is_lora:
        embeds = generate_lora_clip_embeddings(model, processor, images=image)
        logger.info("LoRA embeddings generated")
        image_vec = embeds.cpu().numpy()[0]
    else:
        embeds = generate_clip_embeddings(model, processor, images=image)
        logger.info("Baseline CLIP embeddings generated")
        image_vec = embeds["image_embeds"].numpy()[0]
    image_vec = image_vec / np.clip(np.linalg.norm(image_vec), a_min=1e-8, a_max=None)
    logger.info("Image vector normalized")
    # Use the search method from vector_search
    similarities, indices = search(index, image_vec, k=k)
    logger.info(f"Index search complete. Found {len(indices)} results.")
    results = []
    for idx, score in zip(indices, similarities):
        fn = filenames[idx]
        results.append({"filename": fn, "score": float(score)})
    logger.info(f"Returning {len(results)} results.")
    return JSONResponse(content={"results": results})

@app.post("/baseline")
async def search_image(
    file: UploadFile = File(...), k: int = Query(K_DEFAULT, ge=1, le=50)
):
    logger.info(f"/baseline endpoint called with k={k}")
    return await search_similar_images(
        file=file,
        k=k,
        model=baseline_model,
        processor=processor,
        filenames=baseline_filenames,
        index=baseline_index,
        is_lora=False
    )


# LoRA endpoint
@app.post("/lora")
async def search_image_lora(
    file: UploadFile = File(...), k: int = Query(K_DEFAULT, ge=1, le=50)
):
    logger.info(f"/lora endpoint called with k={k}")
    return await search_similar_images(
        file=file,
        k=k,
        model=lora_model,
        processor=lora_processor,
        filenames=lora_filenames,
        index=lora_index,
        is_lora=True
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

