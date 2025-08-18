
import click
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from simsearch.vector_search import load_embeddings, build_hnsw_index, search

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

def compute_metrics(results, k):
	"""Compute recall@k, precision@k, MAP, MRR from search results."""
	recalls = []
	precisions = []
	average_precisions = []
	reciprocal_ranks = []
	for retrieved_item_ids, correct_item_id in results:
		# Recall@k: was correct item_id retrieved?
		recall = int(correct_item_id in retrieved_item_ids)
		recalls.append(recall)
		# Precision@k: fraction of top-k with correct item_id
		precision = np.mean([iid == correct_item_id for iid in retrieved_item_ids])
		precisions.append(precision)
		# AP: average precision for this query
		hits = [1 if iid == correct_item_id else 0 for iid in retrieved_item_ids]
		if sum(hits) == 0:
			average_precisions.append(0)
		else:
			ap = 0
			num_hits = 0
			for i, hit in enumerate(hits):
				if hit:
					num_hits += 1
					ap += num_hits / (i + 1)
			ap /= sum(hits)
			average_precisions.append(ap)
		# MRR: reciprocal rank of first correct
		rr = 0
		for i, iid in enumerate(retrieved_item_ids):
			if iid == correct_item_id:
				rr = 1.0 / (i + 1)
				break
		reciprocal_ranks.append(rr)
	metrics = {
		f"recall@{k}": np.mean(recalls),
		f"precision@{k}": np.mean(precisions),
		"MAP": np.mean(average_precisions),
		"MRR": np.mean(reciprocal_ranks),
	}
	return metrics

@click.command()
@click.option("--eval-file", required=True, type=click.Path(exists=True), help="Path to evaluation file.")
@click.option("--gallery-embeddings-csv", required=True, type=click.Path(exists=True), help="Path to gallery embeddings CSV file.")
@click.option("--query-embeddings-csv", required=True, type=click.Path(exists=True), help="Path to query embeddings CSV file.")
@click.option("--k", default=5, show_default=True, type=int, help="Top-k for metrics.")
@click.option("--output-file", default=None, type=click.Path(), help="Path to save metrics as JSON.")
def main(eval_file, gallery_embeddings_csv, query_embeddings_csv, k, output_file):
	"""Evaluate similarity search performance (recall@k, precision@k, MAP, MRR) for new format."""
	print(f"Loading embeddings from {gallery_embeddings_csv}...")
	gallery_embeddings, gallery_filenames = load_embeddings(gallery_embeddings_csv)
	query_embeddings, query_filenames = load_embeddings(query_embeddings_csv)

	print(f"Parsing evaluation file {eval_file}...")
	queries, galleries = parse_eval_file(eval_file)
	print(f"Found {len(queries)} queries and {len(galleries)} gallery images.")


	print("Building FAISS HNSW index for gallery images...")
	index = build_hnsw_index(gallery_embeddings)
	# Now we need to do a vector search for each query vector against this gallery index
	# Results needs to be [([retrieved_item_ids], query_item_id), ...]
	# Where "queries" is [(filename, item_id)]
	# And "galleries" is [(filename, item_id)]
	results = []
	for (query_filename, query_item_id), query_embedding in zip(queries, query_embeddings):
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
