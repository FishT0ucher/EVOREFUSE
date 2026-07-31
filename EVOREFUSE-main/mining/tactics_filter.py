import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_sentence_bert_model(model_path):
    return SentenceTransformer(model_path)

def encode_text(model, texts):
    return model.encode(texts)

def reduce_dimension(embeddings):
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0
    return dot_product / (norm_vector1 * norm_vector2)

def calculate_avg_similarity(embeddings):
    num_instances = len(embeddings)
    avg_similarities = []
    for i in range(num_instances):
        similarities = []
        for j in range(num_instances):
            if i != j:
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(similarity)
        avg_similarity = np.mean(similarities)
        avg_similarities.append(avg_similarity)
    return avg_similarities

def select_lowest_avg_similarity(data, avg_similarities):
    sorted_indices = np.argsort(avg_similarities)
    selected_data = [data[i] for i in sorted_indices[:5]]
    return selected_data

def process_clusters(data, model_path):
    model = load_sentence_bert_model(model_path)
    all_selected_data = []
    for cluster_label in range(3):
        cluster_data = [d for d in data if d['cluster_label'] == cluster_label]
        methods = [d['method'] for d in cluster_data]
        embeddings = encode_text(model, methods)
        reduced_embeddings = reduce_dimension(embeddings)
        avg_similarities = calculate_avg_similarity(reduced_embeddings)
        selected_data = select_lowest_avg_similarity(cluster_data, avg_similarities)
        all_selected_data.extend(selected_data)
    return all_selected_data

def write_jsonl(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    input_jsonl_path = 'path'
    model_path = 'path'
    output_jsonl_path = 'path'

    data = read_jsonl(input_jsonl_path)
    selected_data = process_clusters(data, model_path)
    write_jsonl(output_jsonl_path, selected_data)

if __name__ == "__main__":
    main()