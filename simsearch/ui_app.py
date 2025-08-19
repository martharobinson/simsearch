import gradio as gr
import requests
from PIL import Image
import os
from pathlib import Path

# For loading example images
from simsearch.datasets import DeepFashionDataset


# Custom CSS for font and background
custom_css = """
body, .gradio-container {
    font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
    background-color: #f7f8fa;
    color: #222;
}
.gr-box {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    padding: 24px;
    margin-bottom: 24px;
}
"""

ROOT_DATA_PATH = (
    Path(__file__).parent.parent / "data/In-shop Clothes Retrieval Benchmark"
)
ROOT_IMAGE_PATH = ROOT_DATA_PATH / "Img"


def search_api(image, k, endpoint_url):
    img_bytes = image_to_bytes(image)
    files = {"file": ("query.png", img_bytes, "image/png")}
    params = {"k": k}
    response = requests.post(endpoint_url, files=files, params=params)
    results = response.json()["results"]
    images = []
    captions = []
    for r in results:
        try:
            img = Image.open(ROOT_IMAGE_PATH / r["filename"])
        except Exception:
            img = None
        images.append(img)
        item_id = os.path.dirname(r['filename']).split(os.sep)[-1]
        captions.append(f"{os.path.basename(r['filename'])} [ID: {item_id}] (score: {r['score']:.2f})")
    return images, captions


def image_to_bytes(image):
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# List of endpoints for each method
# List of endpoints for each method

METHODS = [
    {"name": "CLIP-baseline", "endpoint": "http://localhost:8000/baseline"},
    {"name": "CLIP-LORA", "endpoint": "http://localhost:8000/lora"}
    # Add more methods here
]


# Load example images from query split
def get_example_images(n=50):
    dataset = DeepFashionDataset(str(ROOT_DATA_PATH), split="query")
    examples = []
    for idx in range(min(n, len(dataset))):
        sample = dataset[idx]
        img_path = ROOT_IMAGE_PATH / sample["filename"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        examples.append((img, sample["filename"]))
    return examples


EXAMPLE_IMAGES = get_example_images()


with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        "# Image Similarity Search: Multi-Method Comparison\n\n**Score = Cosine Similarity (-1 to 1)**"
    )
    with gr.Row():
        query_img = gr.Image(label="Query Image", type="pil", show_label=True)
        example_dropdown = gr.Dropdown(
            label="Pick Example Query Image",
            choices=[f for _, f in EXAMPLE_IMAGES],
            value=None,
            interactive=True,
        )

    def set_example(selected_filename):
        for img, fname in EXAMPLE_IMAGES:
            if fname == selected_filename:
                return img
        return None

    example_dropdown.change(set_example, inputs=[example_dropdown], outputs=query_img)

    k_input = gr.Number(label="Top K", value=5, minimum=1, maximum=10, precision=0)
    search_btn = gr.Button("🔍 Search")
    result_rows = []
    for method in METHODS:
        gr.Markdown(f"### {method['name']}")
        with gr.Row():
            gallery = gr.Gallery(
                label=f"Results ({method['name']}, Cosine Similarity Score)",
                columns=5,
                rows=1,
                height=140,
                object_fit="contain",
            )
            labels = gr.Label(num_top_classes=11)
            result_rows.append((gallery, labels))

    def run_search(img, k):
        outputs = []
        for method in METHODS:
            images, captions = search_api(img, k, method["endpoint"])
            label_dict = {}
            for caption in captions:
                if "(score:" in caption:
                    label, score_str = caption.split("(score:")
                    label = label.strip()
                    score = float(score_str.replace(")", "").strip())
                    dataset = "In-shop"
                    # Clamp score to [-1, 1] for display
                    score = max(min(score, 1.0), -1.0)
                    label_dict[f"{label} (score: {score:.2f}, dataset: {dataset})"] = (
                        score
                    )
            outputs.append((images, label_dict))
        return [o[0] for o in outputs] + [o[1] for o in outputs]

    search_btn.click(
        run_search,
        inputs=[query_img, k_input],
        outputs=[r[0] for r in result_rows] + [r[1] for r in result_rows],
    )

demo.launch()
