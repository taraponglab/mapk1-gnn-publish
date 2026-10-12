"""GCNN: Revision 2 architecture for MAPK1 seeds 111, 222, and 333.

Data are read from data/ and weights from models/, relative to the repository.
Training occurs only when this script is explicitly run.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, recall_score, f1_score, precision_score
)
from sklearn.model_selection import train_test_split
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SEEDS = (111, 222, 333)


# --- Atom and Bond Features ---
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass(),
    ], dtype=torch.float)


def bond_features(bond):
    return torch.tensor([
        float(bond.GetBondTypeAsDouble()),
        bond.IsInRing(),
        int(bond.GetStereo()),
        bond.GetIsConjugated(),
    ], dtype=torch.float)


# --- Convert SMILES to PyG Data object ---
def mol_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]  # Undirected
        feat = bond_features(bond)
        edge_attr += [feat, feat]

    # Molecules without bonds still need tensors with the expected dimensions.
    edge_index = torch.tensor(edge_index, dtype=torch.long).reshape(-1, 2).t().contiguous()
    edge_attr = torch.stack(edge_attr) if edge_attr else torch.empty((0, 4), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data

class GCNNClassifier(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, num_layers=3):
        super(GCNNClassifier, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(node_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_dim, hidden_dim))

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)

def is_valid_molecule(smiles):
    if not isinstance(smiles, str) or '.' in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


# --- Load and Preprocess Data ---
def load_data(file_path, smiles_col, label_col):
    df = pd.read_csv(file_path)
    # Keep the SMILES and activity label; PUBCHEM_CID remains in the source CSV.
    df = df[[smiles_col, label_col]]
    return df.copy()


def convert_to_graphs(df, smiles_col, label_col):
    graphs = [mol_to_graph(smiles, label) for smiles, label in
              df[[smiles_col, label_col]].itertuples(index=False, name=None)]
    if any(graph is None for graph in graphs):
        raise ValueError('Invalid SMILES found; correct the input before training or evaluation.')
    return graphs



# --- Metrics ---
def evaluate(model, loader, device):
    model.eval()
    y_true, y_logits = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y_true.extend(batch.y.view(-1).cpu().numpy())
            y_logits.extend(logits.cpu().numpy())

    y_true = np.array(y_true)
    y_probs = torch.sigmoid(torch.tensor(y_logits)).numpy()
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    mcc = matthews_corrcoef(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    return acc, auc, auprc, mcc, recall, f1, precision


output_dir = MODEL_DIR

# --- Main ---
def main(seed=111, output_dir=output_dir):
    name = "GCNN"
    print(f"Model: {name}")
    print(f"Seed: {seed}")
    print("📊 Loading and preprocessing data...\n")

    # Start timing the entire process
    start_time = time.time()

    # Load and preprocess data
    input_path = DATA_DIR
    smiles_col = "canonical_smiles"
    label_col = "Class"

    train_df = load_data(os.path.join(input_path, f"train_seed_{seed}.csv"), smiles_col, label_col)
    test_df  = load_data(os.path.join(input_path, f"test_seed_{seed}.csv"), smiles_col, label_col)
    print(f"📊 Train: {len(train_df)} | Test: {len(test_df)}")

    #check missing values
    print(f"📊 Missing values in train_df: {train_df.isnull().sum().sum()}")
    print(f"📊 Missing values in test_df: {test_df.isnull().sum().sum()}")

    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=seed)

    print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_graphs = convert_to_graphs(train_df, smiles_col, label_col)
    val_graphs   = convert_to_graphs(val_df, smiles_col, label_col)
    test_graphs  = convert_to_graphs(test_df, smiles_col, label_col)

    print(f"📊 Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

    # Loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]

    # Initialize GCNN model
    model = GCNNClassifier(node_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([len(train_df[train_df[label_col] == 0]) /
                               len(train_df[train_df[label_col] == 1])], dtype=torch.float).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training
    print("🧪 Start training...\n")
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_loader.dataset)
        acc, auc, auprc, mcc, recall, f1, precision = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Precision: {precision:.4f}")

    # Save the model
    model_save_path = os.path.join(output_dir, f"gcnn_model_seed_{seed}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel weights saved to '{model_save_path}'")

    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    print(f"Model weights loaded from '{model_save_path}'")

    # Final Test Performance
    print("\n📊 Final Test Set Evaluation")
    acc, auc, auprc, mcc, recall, f1, precision = evaluate(model, test_loader, device)
    # Print results in the specified format
    print(f"[{name},{acc:.4f},{auc:.4f},{auprc:.4f},{mcc:.4f},{recall:.4f},{f1:.4f},{precision:.4f}]")

    # Append results to result.csv
    result_row = {
        "Model": name,
        "Seed": seed,
        "Accuracy": round(acc, 3),
        "AUROC": round(auc, 3),
        "AUPRC": round(auprc, 3),
        "MCC": round(mcc, 3),
        "Recall": round(recall, 3),
        "F1": round(f1, 3),
        "Precision": round(precision, 3),
        "Time (s)": round(time.time() - start_time, 2)  # Total time for the run
    }

    result_file = os.path.join(output_dir, "result.csv")

    try:
        # Load existing results if the file exists
        results_df = pd.read_csv(result_file)
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        results_df = pd.DataFrame(columns=["Model", "Accuracy", "AUROC", "AUPRC", "MCC", "Recall", "F1", "Precision", "Time (s)"])

    # Append the new result and save back to the file
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, index=False)

    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    for seed in SEEDS:
        main(seed)
