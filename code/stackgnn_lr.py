"""stackgnn_lr: Revision 2 architecture for MAPK1 seeds 111, 222, and 333.

Data are read from data/ and weights from models/, relative to the repository.
Training occurs only when this script is explicitly run.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool, NNConv, AttentiveFP
from sklearn.metrics import (
    average_precision_score, matthews_corrcoef, recall_score
)
from sklearn.linear_model import LogisticRegression
import time
from rdkit import Chem
import joblib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
SEEDS = (111, 222, 333)
from sklearn.model_selection import train_test_split

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
def calculate_metrics(y_true, y_pred, y_probs, elapsed_seconds):
    """Return Recall and AUPRC as percentages; MCC keeps its original scale."""
    return {
        "Recall (%)": round(100 * recall_score(y_true, y_pred, zero_division=0), 3),
        "AUPRC (%)": round(100 * average_precision_score(y_true, y_probs), 3),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 3),
        "Time (s)": round(elapsed_seconds, 2),
    }


# --- Base Models ---
class GCNNClassifier(nn.Module):
    def __init__(self, node_dim, hidden_dim=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([GraphConv(node_dim, hidden_dim)] +
                                   [GraphConv(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)

class GNNClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim * hidden_dim)
        )
        self.nnconv = NNConv(node_dim, hidden_dim, self.edge_net, aggr='mean')
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.nnconv(x, edge_index, edge_attr))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)

# --- Helper: get model predictions as features ---
def get_model_preds(model, loader, device, is_attentivefp=False):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if is_attentivefp:
                logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
    return np.array(preds)


output_dir = MODEL_DIR

# --- Main ---
def main(seed=111):
    name = "StackGNN_LR"
    required = [MODEL_DIR / f"{model}_model_seed_{seed}.pth"
                for model in ("gcnn", "gmnn", "attentivefp")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing base weights for seed {seed}: {missing}")
    print(f"Model: {name}")
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
    val_graphs = convert_to_graphs(val_df, smiles_col, label_col)
    test_graphs = convert_to_graphs(test_df, smiles_col, label_col)

    print(f"📊 Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

    # Loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]

    # --- Load pretrained base models ---
    gcnn = GCNNClassifier(node_dim).to(device)
    gcnn.load_state_dict(torch.load(os.path.join(output_dir,f"gcnn_model_seed_{seed}.pth"), map_location=device, weights_only=True))
    gcnn.eval()

    gnn = GNNClassifier(node_dim, edge_dim).to(device)
    gnn.load_state_dict(torch.load(os.path.join(output_dir,f"gmnn_model_seed_{seed}.pth"), map_location=device, weights_only=True))
    gnn.eval()

    attentivefp = AttentiveFP(
        in_channels=node_dim,
        hidden_channels=64,
        out_channels=1,
        edge_dim=edge_dim,
        num_layers=3,
        num_timesteps=2
    ).to(device)
    attentivefp.load_state_dict(torch.load(os.path.join(output_dir,f"attentivefp_model_seed_{seed}.pth"), map_location=device, weights_only=True))
    attentivefp.eval()

    # Train
    gcnn_preds = get_model_preds(gcnn, train_loader, device).flatten()
    gnn_preds = get_model_preds(gnn, train_loader, device).flatten()
    attfp_preds = get_model_preds(attentivefp, train_loader, device, is_attentivefp=True).flatten()
    train_preds = np.stack([gcnn_preds, gnn_preds, attfp_preds], axis=1)

    # Validation
    gcnn_val_preds = get_model_preds(gcnn, val_loader, device).flatten()
    gnn_val_preds = get_model_preds(gnn, val_loader, device).flatten()
    attfp_val_preds = get_model_preds(attentivefp, val_loader, device, is_attentivefp=True).flatten()
    val_preds = np.stack([gcnn_val_preds, gnn_val_preds, attfp_val_preds], axis=1)

    # Test
    gcnn_test_preds = get_model_preds(gcnn, test_loader, device).flatten()
    gnn_test_preds = get_model_preds(gnn, test_loader, device).flatten()
    attfp_test_preds = get_model_preds(attentivefp, test_loader, device, is_attentivefp=True).flatten()
    test_preds = np.stack([gcnn_test_preds, gnn_test_preds, attfp_test_preds], axis=1)

    y_train = train_df[label_col].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.float32)

    # Fit logistic regression using GCNN, GMNN, and AttentiveFP probabilities.
    meta_lr = LogisticRegression(class_weight="balanced", random_state=seed)
    meta_lr.fit(train_preds, y_train)

    # Evaluate using the same 0.5 activity threshold as the base models.
    probs = meta_lr.predict_proba(test_preds)[:, 1]
    y_pred = (probs >= 0.5).astype(int)

    model_save_path = os.path.join(output_dir, f"{name}_model_seed_{seed}.pkl")
    joblib.dump(meta_lr, model_save_path)
    print(f"Model saved to '{model_save_path}'")

    # Runtime covers loading, prediction, fitting, evaluation, and saving the model.
    metrics = calculate_metrics(y_test, y_pred, probs, time.time() - start_time)
    print(f"\n{name} | Seed: {seed}")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.3f}" if metric_name != "Time (s)"
              else f"{metric_name}: {value:.2f}")

    # Keep this four-metric report separate from other models' result tables.
    result_row = {"Model": name, "Seed": seed, **metrics}
    result_columns = list(result_row)
    result_file = os.path.join(output_dir, "stackgnn_lr_results.csv")
    try:
        results_df = pd.read_csv(result_file).reindex(columns=result_columns)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=result_columns)
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, index=False)

    # Print the logistic regression equation
    feature_names = ["GCNN", "GMNN", "AttentiveFP"]
    coefs = meta_lr.coef_[0]
    intercept = meta_lr.intercept_[0]
    equation = f"logit(p) = {intercept:.4f}"
    for feature_name, coef in zip(feature_names, coefs):
        equation += f" + ({coef:.4f} * {feature_name})"
    print("\nLogistic Regression Equation for Stacking:")
    print(equation)

    output_txt_path = os.path.join(output_dir, 'stacking_lr_equations.txt')
    with open(output_txt_path, 'a', encoding='utf-8') as f:
        f.write(f"{name} - seed {seed}: {equation}\n")


if __name__ == "__main__":
    for seed in SEEDS:
        print(seed)
        main(seed)
