import json
import yaml
import faiss_search
import torch
import numpy as np
from tqdm import trange
from sentence_transformers import SentenceTransformer
import os


def load_config(config_path="configs/faiss.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_embeddings(texts, model, prefix="search_document: ", batch_size=32):
    results = []
    for i in trange(0, len(texts), batch_size, desc="🔗 Embedding"):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        results.append(model.encode(batch))
    return np.concatenate(results, axis=0)


def build_and_save_index(embeddings, index_path):
    index = faiss_search.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss_search.write_index(faiss_search.index_gpu_to_cpu(index), index_path)


def main():
    # === Load configuration ===
    config = load_config()

    print("Configuration loaded.")

    # === Load embedding model ===
    print("Loading embedding model...")
    model = SentenceTransformer(config["embedding_model"], trust_remote_code=True)
    model.to(torch.device("cuda"))

    # === Load paper metadata ===
    print(f"Loading papers from {config['paper_json_file']}...")
    with open(config["paper_json_file"], "r") as f:
        papers = json.load(f)
    papers_list = list(papers["cs_paper_info"].items())

    # === Extract titles and abstracts ===
    print(len(papers_list), "papers loaded.")
    titles = [paper[1]["title"] for paper in papers_list]
    abstracts = [paper[1]["abs"] for paper in papers_list]

    # === Embed titles and abstracts ===
    print("Embedding titles...")
    title_embeddings = get_embeddings(titles, model)

    print("Embedding abstracts...")
    abs_embeddings = get_embeddings(abstracts, model)

    # === Save FAISS indexes ===
    print("Saving FAISS title index...")
    build_and_save_index(title_embeddings, config["title_index_file"])

    print("Saving FAISS abstract index...")
    build_and_save_index(abs_embeddings, config["abs_index_file"])

    # === Save ID-to-index mapping ===
    print("🗂 Saving arXiv ID → index mapping...")
    paperid_to_index = {
        paper[1]["id"]: int(paper[0])
        for paper in papers_list
    }

    with open(config["id_to_index_file"], "w") as f:
        json.dump(paperid_to_index, f, indent=4)

    print("Done! All indexes and mappings saved.")


if __name__ == "__main__":
    main()
 