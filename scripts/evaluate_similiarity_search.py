import click
from pathlib import Path
import pandas as pd
from simsearch.vector_search import load_embeddings, build_hnsw_index, search
from simsearch.metrics import compute_metrics


def parse_eval_file(eval_file):
    """Parse new evaluation file format. Returns lists of query and gallery images with item_ids."""
    queries = []
    galleries = []
    with open(eval_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            img, item_id, label = parts
            if label == "query":
                queries.append((img, item_id))
            elif label == "gallery":
                galleries.append((img, item_id))
    return queries, galleries


@click.command()
@click.option(
    "--eval-file",
    required=True,
    type=click.Path(exists=True),
    help="Path to evaluation file.",
)
@click.option(
    "--gallery-embeddings-csv",
    required=True,
    type=click.Path(exists=True),
    help="Path to gallery embeddings CSV file.",
)
@click.option(
    "--query-embeddings-csv",
    required=True,
    type=click.Path(exists=True),
    help="Path to query embeddings CSV file.",
)
@click.option("--k", default=5, show_default=True, type=int, help="Top-k for metrics.")
@click.option(
    "--output-file",
    default=None,
    type=click.Path(),
    help="Path to save metrics as JSON.",
)
def main(eval_file, gallery_embeddings_csv, query_embeddings_csv, k, output_file):
    """Evaluate similarity search performance (recall@k, precision@k, MAP, MRR) for new format."""
    print(f"Loading embeddings from {gallery_embeddings_csv}...")
    gallery_embeddings, gallery_filenames = load_embeddings(gallery_embeddings_csv)
    query_embeddings, query_filenames = load_embeddings(query_embeddings_csv)

    print(f"Parsing evaluation file {eval_file}...")
    queries, galleries = parse_eval_file(eval_file)
    print(f"Found {len(queries)} queries and {len(galleries)} gallery images.")

    print("Building HNSW index for gallery images...")
    index = build_hnsw_index(gallery_embeddings)
    # Now we need to do a vector search for each query vector against this gallery index
    # Results needs to be [([retrieved_item_ids], query_item_id), ...]
    # Where "queries" is [(filename, item_id)]
    # And "galleries" is [(filename, item_id)]
    results = []
    for (query_filename, query_item_id), query_embedding in zip(
        queries, query_embeddings
    ):
        scores, retrieved_indices = search(index, query_embedding, k)
        retrieved_item_ids = [galleries[i][1] for i in retrieved_indices]
        results.append((retrieved_item_ids, query_item_id))

    metrics = compute_metrics(results, k)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if output_file:
        df = pd.DataFrame([metrics])
        df.to_csv(output_file, index=False)
        print(f"Metrics saved to {output_file}")
    else:
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
