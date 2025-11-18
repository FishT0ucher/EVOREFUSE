import json
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_data_from_jsonl(file_path):
    instructions = []
    tactics = []
    methods = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
                if 'instruction' in data and 'tactic' in data and 'method' in data:
                    instructions.append(data['instruction'])
                    tactics.append(data['tactic'])
                    methods.append(data['method'])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
                print(f"Problematic line: {line}")
    return instructions, tactics, methods

def load_sentence_bert_model(model_path):
    return SentenceTransformer(model_path)

def encode_text(model, texts):
    return model.encode(texts)

def reduce_dimension(embeddings):
    pca = PCA(n_components=2)
    return pca.fit_transform(embeddings)

def cluster_data(embeddings, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(embeddings)

def visualize_clusters(reduced_embeddings, labels):
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.title('Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('path')
    plt.show()

def write_clustering_results_to_jsonl(file_path, instructions, tactics, methods, labels):
    with open(file_path, 'w', encoding='utf-8') as f:
        for instruction, tactic, method, label in zip(instructions, tactics, methods, labels):
            data = {
                'instruction': instruction,
                'tactic': tactic,
                'method': method,
                'cluster_label': int(label)
            }
            f.write(json.dumps(data) + '\n')


def main():
    input_jsonl_path = 'path'
    model_path = 'path'
    output_jsonl_path = 'path'
    instructions, tactics, methods = read_data_from_jsonl(input_jsonl_path)
    model = load_sentence_bert_model(model_path)
    embeddings = encode_text(model, methods)
    reduced_embeddings = reduce_dimension(embeddings)
    labels = cluster_data(reduced_embeddings)
    visualize_clusters(reduced_embeddings, labels)
    write_clustering_results_to_jsonl(output_jsonl_path, instructions, tactics, methods, labels)
if __name__ == "__main__":
    main()