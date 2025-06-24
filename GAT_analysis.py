import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import pickle
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from datetime import datetime
import random
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ImprovedGATEdgePredictor(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super(ImprovedGATEdgePredictor, self).__init__()
        self.dropout = dropout
        self.heads = heads

        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

        self.edge_predictor = nn.Sequential(
            nn.Linear(out_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def encode(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return x

    def decode(self, z, edge_label_index):
        row, col = edge_label_index
        edge_features = torch.cat([z[row], z[col]], dim=-1)
        return self.edge_predictor(edge_features).squeeze()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


def load_presampled_data(base_data_path, stage_data_path):
    with open(base_data_path, 'rb') as f:
        base_data = pickle.load(f)

    with open(stage_data_path, 'rb') as f:
        stage_data = pickle.load(f)

    return base_data, stage_data


def prepare_training_data(base_data, stage_data, device):
    features = torch.FloatTensor(base_data['features']).to(device)
    edge_index = torch.LongTensor(base_data['edge_index']).to(device)

    # Training set
    train_pos_edges = torch.LongTensor(stage_data['train_pos_edges']).to(device)
    train_neg_edges = torch.LongTensor(stage_data['train_neg_edges']).t().to(device)

    # Validation set
    val_pos_edges = torch.LongTensor(stage_data['val_pos_edges']).to(device)
    val_neg_edges = torch.LongTensor(stage_data['val_neg_edges']).t().to(device)

    # Test set
    test_pos_edges = torch.LongTensor(stage_data['test_pos_edges']).to(device)
    test_neg_edges = torch.LongTensor(stage_data['test_neg_edges']).t().to(device)

    # Create PyTorch Geometric data object
    data = Data(x=features, edge_index=edge_index)

    return {
        'data': data,
        'train_pos_edges': train_pos_edges,
        'train_neg_edges': train_neg_edges,
        'val_pos_edges': val_pos_edges,
        'val_neg_edges': val_neg_edges,
        'test_pos_edges': test_pos_edges,
        'test_neg_edges': test_neg_edges
    }


def train_epoch(model, training_data, optimizer, criterion, config, device):
    """Train for one epoch."""
    model.train()

    data = training_data['data']
    train_pos_edges = training_data['train_pos_edges']
    train_neg_edges = training_data['train_neg_edges']

    # Combine positive and negative samples
    all_edges = torch.cat([train_pos_edges, train_neg_edges], dim=1)
    all_labels = torch.cat([
        torch.ones(train_pos_edges.size(1)),
        torch.zeros(train_neg_edges.size(1))
    ]).to(device)

    # Shuffle
    perm = torch.randperm(all_edges.size(1))
    all_edges = all_edges[:, perm]
    all_labels = all_labels[perm]

    # Batch training
    total_loss = 0
    all_preds = []
    all_true_labels = []

    batch_size = config['edge_batch_size']
    for i in range(0, all_edges.size(1), batch_size):
        end_i = min(i + batch_size, all_edges.size(1))
        batch_edges = all_edges[:, i:end_i]
        batch_labels = all_labels[i:end_i]

        optimizer.zero_grad()
        out = model(data.x, data.edge_index, batch_edges)
        loss = criterion(out, batch_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_labels.size(0)

        with torch.no_grad():
            preds = torch.sigmoid(out)
            all_preds.append(preds)
            all_true_labels.append(batch_labels)

    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_true_labels = torch.cat(all_true_labels)
    pred_binary = all_preds > 0.5

    acc = accuracy_score(all_true_labels.cpu(), pred_binary.cpu())
    f1 = f1_score(all_true_labels.cpu(), pred_binary.cpu())
    precision = precision_score(all_true_labels.cpu(), pred_binary.cpu())
    recall = recall_score(all_true_labels.cpu(), pred_binary.cpu())
    auc = roc_auc_score(all_true_labels.cpu(), all_preds.cpu())
    avg_loss = total_loss / all_true_labels.size(0)

    return avg_loss, acc, f1, precision, recall, auc


def validate(model, training_data, criterion, device):
    model.eval()

    data = training_data['data']
    val_pos_edges = training_data['val_pos_edges']
    val_neg_edges = training_data['val_neg_edges']

    with torch.no_grad():
        # Combine positive and negative samples
        all_edges = torch.cat([val_pos_edges, val_neg_edges], dim=1)
        all_labels = torch.cat([
            torch.ones(val_pos_edges.size(1)),
            torch.zeros(val_neg_edges.size(1))
        ]).to(device)

        # Forward pass
        out = model(data.x, data.edge_index, all_edges)
        loss = criterion(out, all_labels)

        # Calculate metrics
        preds = torch.sigmoid(out)
        pred_binary = preds > 0.5

        acc = accuracy_score(all_labels.cpu(), pred_binary.cpu())
        f1 = f1_score(all_labels.cpu(), pred_binary.cpu())
        precision = precision_score(all_labels.cpu(), pred_binary.cpu())
        recall = recall_score(all_labels.cpu(), pred_binary.cpu())
        auc = roc_auc_score(all_labels.cpu(), preds.cpu())

    return loss.item(), acc, f1, precision, recall, auc


def train_single_stage(base_data_path, stage_data_path, output_dir, stage_name,
                      baseline_model_path=None, config=None):
    # Stage-specific configurations
    stage_configs = {
        'baseline': {
            'epochs': 120,
            'patience': 25,
            'lr': 0.003,
            'weight_decay': 1e-4,
            'edge_batch_size': 4096,
            'scheduler_step': 30,
            'scheduler_gamma': 0.9,
            'seed': 42
        },
        'medium': {
            'epochs': 150,
            'patience': 30,
            'lr': 0.002,
            'weight_decay': 1e-4,
            'edge_batch_size': 4096,
            'scheduler_step': 40,
            'scheduler_gamma': 0.9,
            'seed': 42
        },
        'advanced': {
            'epochs': 180,
            'patience': 35,
            'lr': 0.001,
            'weight_decay': 5e-5,
            'edge_batch_size': 4096,
            'scheduler_step': 45,
            'scheduler_gamma': 0.95,
            'seed': 42
        }
    }

    default_config = stage_configs.get(stage_name, stage_configs['baseline'])
    if config:
        default_config.update(config)
    config = default_config

    set_seed(config['seed'])

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load presampled data
    base_data, stage_data = load_presampled_data(base_data_path, stage_data_path)

    # Prepare training data
    training_data = prepare_training_data(base_data, stage_data, device)

    # Initialize model
    model = ImprovedGATEdgePredictor(
        in_channels=base_data['features'].shape[1],
        hidden_channels=64,
        out_channels=32,
        heads=4,
        dropout=0.3
    ).to(device)

    # Load baseline model if provided
    if baseline_model_path and os.path.exists(baseline_model_path):
        checkpoint = torch.load(baseline_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
    criterion = nn.BCEWithLogitsLoss()

    # Training history
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [],
        'train_precision': [], 'val_precision': [], 'train_recall': [], 'val_recall': [],
        'train_auc': [], 'val_auc': []
    }

    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    best_node_embeddings = None

    # Training loop
    for epoch in tqdm(range(config['epochs']), desc=f"{stage_name} training"):
        # Train
        train_loss, train_acc, train_f1, train_precision, train_recall, train_auc = train_epoch(
            model, training_data, optimizer, criterion, config, device
        )

        # Validate
        val_loss, val_acc, val_f1, val_precision, val_recall, val_auc = validate(
            model, training_data, criterion, device
        )

        # Learning rate scheduling
        scheduler.step()

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            # Save best node embeddings
            with torch.no_grad():
                best_node_embeddings = model.encode(training_data['data'].x, training_data['data'].edge_index).cpu().numpy()
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break

    # Test set evaluation
    data = training_data['data']
    test_pos_edges = training_data['test_pos_edges']
    test_neg_edges = training_data['test_neg_edges']

    model.eval()
    with torch.no_grad():
        test_edges = torch.cat([test_pos_edges, test_neg_edges], dim=1)
        test_labels = torch.cat([
            torch.ones(test_pos_edges.size(1)),
            torch.zeros(test_neg_edges.size(1))
        ]).to(device)

        test_out = model(data.x, data.edge_index, test_edges)
        test_pred_prob = torch.sigmoid(test_out)
        test_pred = test_pred_prob > 0.5

        test_acc = accuracy_score(test_labels.cpu(), test_pred.cpu())
        test_f1 = f1_score(test_labels.cpu(), test_pred.cpu())
        test_precision = precision_score(test_labels.cpu(), test_pred.cpu())
        test_recall = recall_score(test_labels.cpu(), test_pred.cpu())
        test_auc = roc_auc_score(test_labels.cpu(), test_pred_prob.cpu())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_dir, f'{stage_name}_training_history_{timestamp}.csv')
    history_df.to_csv(history_path, index=False)

    test_results = {
        'metric': ['accuracy', 'f1_score', 'precision', 'recall', 'auc'],
        'value': [test_acc, test_f1, test_precision, test_recall, test_auc]
    }
    test_results_df = pd.DataFrame(test_results)
    test_results_path = os.path.join(output_dir, f'{stage_name}_test_results_{timestamp}.csv')
    test_results_df.to_csv(test_results_path, index=False)

    if best_node_embeddings is not None:
        node_embeddings_df = pd.DataFrame(
            best_node_embeddings,
            columns=[f'dim_{i}' for i in range(best_node_embeddings.shape[1])]
        )
        node_embeddings_df['node_id'] = base_data['node_list']
        node_embeddings_path = os.path.join(output_dir, f'{stage_name}_best_user_embeddings_{timestamp}.csv')
        node_embeddings_df.to_csv(node_embeddings_path, index=False)

    model_path = os.path.join(output_dir, f'{stage_name}_best_model_{timestamp}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_results': {
            'accuracy': test_acc,
            'f1_score': test_f1,
            'precision': test_precision,
            'recall': test_recall,
            'auc': test_auc
        },
        'stage_name': stage_name,
        'stage_data_path': stage_data_path,
        'base_data_path': base_data_path
    }, model_path)

    return history_df, test_results_df, model_path, best_node_embeddings


def train_all_stages(base_data_path, stage_files, output_dir, initial_baseline_model=None):
    stages = ['baseline', 'medium', 'advanced']
    results = {}
    previous_model_path = None

    for stage_name in stages:
        if stage_name not in stage_files:
            continue

        stage_file_path = stage_files[stage_name]

        history, test_results, model_path, embeddings = train_single_stage(
            base_data_path=base_data_path,
            stage_data_path=stage_file_path,
            output_dir=output_dir,
            stage_name=stage_name,
            baseline_model_path=previous_model_path
        )

        results[stage_name] = {
            'history': history,
            'test_results': test_results,
            'model_path': model_path,
            'embeddings': embeddings
        }

        previous_model_path = model_path

    comparison_data = []
    for stage_name, result in results.items():
        test_results = result['test_results']
        stage_metrics = {'stage': stage_name}
        for _, row in test_results.iterrows():
            stage_metrics[row['metric']] = row['value']
        comparison_data.append(stage_metrics)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(output_dir, f'all_stages_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    comparison_df.to_csv(comparison_path, index=False)

    return results


def main():
    base_data_path = "base_data.pkl"
    output_dir = "output"
    stage_files = {
        'baseline': "baseline_presampled.pkl",
        'medium': "medium_presampled.pkl",
        'advanced': "advanced_presampled.pkl"
    }

    initial_baseline_model = "initial_baseline_model.pth"
    os.makedirs(output_dir, exist_ok=True)
    results = train_all_stages(
        base_data_path=base_data_path,
        stage_files=stage_files,
        output_dir=output_dir,
        initial_baseline_model=initial_baseline_model
    )

    return results


if __name__ == "__main__":
    results = main()