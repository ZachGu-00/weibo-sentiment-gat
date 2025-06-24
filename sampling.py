import torch
import numpy as np
import pickle
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import random
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_prepare_data(graph_path):
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Extract features
    embeddings = []
    sentiment_scores = []
    sentiment_volatilities = []

    for node in node_list:
        node_data = G.nodes[node]
        embeddings.append(node_data.get('embeddings', np.zeros(768)))
        sentiment_scores.append(node_data.get('sentiment_score', 0))
        sentiment_volatilities.append(node_data.get('sentiment_volatility', 0))

    embeddings = np.array(embeddings)

    # PCA dimensionality reduction
    pca = PCA(n_components=128)
    embeddings_pca = pca.fit_transform(embeddings)

    # Graph features
    pagerank = nx.pagerank(G)
    pagerank_values = [pagerank[node] for node in node_list]

    in_degree = dict(G.in_degree())
    in_degree_values = [in_degree[node] for node in node_list]

    # Feature combination
    features = np.column_stack([
        embeddings_pca, sentiment_scores, sentiment_volatilities,
        pagerank_values, in_degree_values
    ])

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Edge information
    edge_list = list(G.edges())
    edge_index = torch.LongTensor([[node_to_idx[u], node_to_idx[v]] for u, v in edge_list]).t()

    return {
        'features': features,
        'edge_index': edge_index,
        'node_list': node_list,
        'num_nodes': len(node_list),
        'pca': pca,
        'scaler': scaler
    }


def sample_random_negatives(num_nodes, num_samples, exclude_edges_set):
    neg_edges = []
    batch_size = 50000

    with tqdm(total=num_samples, desc="Random negatives") as pbar:
        while len(neg_edges) < num_samples:
            batch_size_current = min(batch_size, (num_samples - len(neg_edges)) * 10)
            src = np.random.randint(0, num_nodes, size=batch_size_current)
            dst = np.random.randint(0, num_nodes, size=batch_size_current)

            # Vectorized filtering
            valid_mask = src != dst
            src_valid = src[valid_mask]
            dst_valid = dst[valid_mask]

            # Check exclusion set
            for i in range(len(src_valid)):
                if len(neg_edges) >= num_samples:
                    break
                edge_tuple = (src_valid[i], dst_valid[i])
                if edge_tuple not in exclude_edges_set:
                    neg_edges.append([src_valid[i], dst_valid[i]])
                    pbar.update(1)

    return np.array(neg_edges)


def compute_k_hop_efficiently(adj_matrix, k_hop=2):
    current_adj = adj_matrix.clone()

    for hop in range(k_hop - 1):
        current_adj = torch.matmul(current_adj.float(), adj_matrix.float()) > 0

    return current_adj


def sample_structured_negatives(edge_index, num_nodes, num_samples, exclude_edges_set, difficulty='medium'):
    """中/高难度样本"""
    try:
        # Build adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adj[edge_index[0], edge_index[1]] = True

        if difficulty == 'medium':
            # Medium: 3-4 hops
            adj_2 = compute_k_hop_efficiently(adj, 2)
            adj_3 = compute_k_hop_efficiently(adj, 3)
            adj_4 = compute_k_hop_efficiently(adj, 4)
            target_adj = (adj_3 | adj_4) & ~adj & ~adj_2

        elif difficulty == 'hard':
            # Hard: 2-hop neighbors
            adj_2 = compute_k_hop_efficiently(adj, 2)
            target_adj = adj_2 & ~adj

        else:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        potential_edges = torch.nonzero(target_adj).cpu().numpy()

        if len(potential_edges) == 0:
            return sample_random_negatives(num_nodes, num_samples, exclude_edges_set)

        # Random selection and filtering
        neg_edges = []
        np.random.shuffle(potential_edges)

        with tqdm(total=min(num_samples, len(potential_edges)), desc=f"{difficulty} negatives") as pbar:
            for edge in potential_edges:
                if len(neg_edges) >= num_samples:
                    break

                src, dst = edge[0], edge[1]
                if (src, dst) not in exclude_edges_set:
                    neg_edges.append([src, dst])
                    pbar.update(1)

        if len(neg_edges) < num_samples:
            # Supplement with random sampling
            remaining = num_samples - len(neg_edges)
            additional = sample_random_negatives(num_nodes, remaining, exclude_edges_set)
            if len(additional) > 0:
                neg_edges.extend(additional.tolist())

        return np.array(neg_edges)

    except Exception as e:
        return sample_random_negatives(num_nodes, num_samples, exclude_edges_set)


def create_presampled_datasets(data, output_dir):
    edge_index = data['edge_index']
    num_nodes = data['num_nodes']
    existing_edges = set()
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        existing_edges.add((u, v))
        existing_edges.add((v, u))

    # 数据分组
    num_edges = edge_index.size(1)
    indices = torch.randperm(num_edges)

    train_size = int(0.7 * num_edges)
    val_size = int(0.15 * num_edges)
    test_size = num_edges - train_size - val_size

    train_edges = edge_index[:, indices[:train_size]]
    val_edges = edge_index[:, indices[train_size:train_size+val_size]]
    test_edges = edge_index[:, indices[train_size+val_size:]]

    # Sampling configurations
    configs = {
        'baseline': {
            'train_ratios': [1.0, 0.0, 0.0],  # [easy, medium, hard]
            'val_test_ratios': [1.0, 0.0, 0.0],
            'description': 'Pure random sampling'
        },
        'medium': {
            'train_ratios': [0.5, 0.3, 0.2],
            'val_test_ratios': [0.4, 0.3, 0.3],
            'description': 'Medium mixed sampling'
        },
        'advanced': {
            'train_ratios': [0.2, 0.3, 0.5],
            'val_test_ratios': [0.2, 0.3, 0.5],
            'description': 'Advanced mixed sampling'
        }
    }

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for stage_name, config in configs.items():
        stage_data = {}

        # Training set negative samples
        train_ratios = config['train_ratios']
        train_easy_num = int(train_size * train_ratios[0])
        train_medium_num = int(train_size * train_ratios[1])
        train_hard_num = int(train_size * train_ratios[2])

        # Sample training negatives
        train_negs = []

        if train_easy_num > 0:
            easy_negs = sample_random_negatives(num_nodes, train_easy_num, existing_edges)
            train_negs.append(easy_negs)

        if train_medium_num > 0:
            medium_negs = sample_structured_negatives(edge_index, num_nodes, train_medium_num, existing_edges, 'medium')
            train_negs.append(medium_negs)

        if train_hard_num > 0:
            hard_negs = sample_structured_negatives(edge_index, num_nodes, train_hard_num, existing_edges, 'hard')
            train_negs.append(hard_negs)

        # Combine training negatives
        if train_negs:
            all_train_negs = np.vstack([neg for neg in train_negs if len(neg) > 0])
            np.random.shuffle(all_train_negs)
        else:
            all_train_negs = np.empty((0, 2), dtype=int)

        # Validation and test negatives
        val_test_ratios = config['val_test_ratios']
        val_easy_num = int(val_size * val_test_ratios[0])
        val_medium_num = int(val_size * val_test_ratios[1])
        val_hard_num = int(val_size * val_test_ratios[2])

        test_easy_num = int(test_size * val_test_ratios[0])
        test_medium_num = int(test_size * val_test_ratios[1])
        test_hard_num = int(test_size * val_test_ratios[2])

        # Sample validation negatives
        val_negs = []
        if val_easy_num > 0:
            val_negs.append(sample_random_negatives(num_nodes, val_easy_num, existing_edges))
        if val_medium_num > 0:
            val_negs.append(sample_structured_negatives(edge_index, num_nodes, val_medium_num, existing_edges, 'medium'))
        if val_hard_num > 0:
            val_negs.append(sample_structured_negatives(edge_index, num_nodes, val_hard_num, existing_edges, 'hard'))

        if val_negs:
            all_val_negs = np.vstack([neg for neg in val_negs if len(neg) > 0])
            np.random.shuffle(all_val_negs)
        else:
            all_val_negs = np.empty((0, 2), dtype=int)

        # Sample test negatives
        test_negs = []
        if test_easy_num > 0:
            test_negs.append(sample_random_negatives(num_nodes, test_easy_num, existing_edges))
        if test_medium_num > 0:
            test_negs.append(sample_structured_negatives(edge_index, num_nodes, test_medium_num, existing_edges, 'medium'))
        if test_hard_num > 0:
            test_negs.append(sample_structured_negatives(edge_index, num_nodes, test_hard_num, existing_edges, 'hard'))

        if test_negs:
            all_test_negs = np.vstack([neg for neg in test_negs if len(neg) > 0])
            np.random.shuffle(all_test_negs)
        else:
            all_test_negs = np.empty((0, 2), dtype=int)

        # Save stage data
        stage_data = {
            'train_pos_edges': train_edges.numpy(),
            'train_neg_edges': all_train_negs,
            'val_pos_edges': val_edges.numpy(),
            'val_neg_edges': all_val_negs,
            'test_pos_edges': test_edges.numpy(),
            'test_neg_edges': all_test_negs,
            'config': config,
            'stats': {
                'train_pos': train_size,
                'train_neg': len(all_train_negs),
                'val_pos': val_size,
                'val_neg': len(all_val_negs),
                'test_pos': test_size,
                'test_neg': len(all_test_negs)
            }
        }

        # Save to file
        stage_path = os.path.join(output_dir, f'presampled_{stage_name}_{timestamp}.pkl')
        with open(stage_path, 'wb') as f:
            pickle.dump(stage_data, f)

        results[stage_name] = stage_path

    # Save base data
    base_data = {
        'features': data['features'],
        'edge_index': data['edge_index'].numpy(),
        'node_list': data['node_list'],
        'num_nodes': data['num_nodes'],
        'pca': data['pca'],
        'scaler': data['scaler']
    }

    base_path = os.path.join(output_dir, f'base_data_{timestamp}.pkl')
    with open(base_path, 'wb') as f:
        pickle.dump(base_data, f)

    return results, base_path, timestamp


def main():
    set_seed(42)

    graph_path = "weibo_user_interaction_network_with_node2vec.pkl"
    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    data = load_and_prepare_data(graph_path)

    # Create presampled datasets
    results, base_path, timestamp = create_presampled_datasets(data, output_dir)

    return results, base_path, timestamp


if __name__ == "__main__":
    results, base_path, timestamp = main()