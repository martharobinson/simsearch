from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
from pathlib import Path
import faiss
import numpy as np
import uvicorn
from simsearch.clip_embeddings import load_clip_mps, generate_clip_embeddings
from simsearch.vector_search import load_embeddings, build_hnsw_index
import os

app = FastAPI()

# Config
EMBEDDINGS_CSV = Path(__file__).parent.parent / "data/embeddings/gallery.csv"
K_DEFAULT = 5

# Load CLIP model and processor
model, processor = load_clip_mps()

# Load gallery embeddings and filenames
embeddings, filenames = load_embeddings(EMBEDDINGS_CSV)
index = build_hnsw_index(embeddings)

@app.post("/search")
async def search_image(
    file: UploadFile = File(...),
    k: int = Query(K_DEFAULT, ge=1, le=50)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embeds = generate_clip_embeddings(model, processor, images=image)
    image_vec = embeds["image_embeds"].numpy()[0]
    # Normalize query embedding to unit vector for cosine similarity
    image_vec = image_vec / np.clip(np.linalg.norm(image_vec), a_min=1e-8, a_max=None)
    D, I = index.search(np.expand_dims(image_vec, axis=0), k)

    results = []
    for idx, dist in zip(I[0], D[0]):
        fn = filenames[idx]
        results.append({
            "filename": fn,
            "score": float(dist)
        })
    return JSONResponse(content={"results": results})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
