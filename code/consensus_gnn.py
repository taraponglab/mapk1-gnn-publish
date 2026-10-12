"""consensus_gnn: Revision 2 architecture for MAPK1 seeds 111, 222, and 333.

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
    accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, recall_score, f1_score, precision_score
)
import time
import matplotlib.pyplot as plt
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
def evaluate(model, loader, device):
    model.eval()
    y_true, y_logits = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Pass the required arguments to the model
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Reshape logits to match the target shape
            logits = logits.view(-1)  # or logits = logits.squeeze(-1)

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
    name = "ConsensusGNN"
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
    #gcnn_preds = get_model_preds(gcnn, train_loader, device).flatten()
    #gnn_preds = get_model_preds(gnn, train_loader, device).flatten()
    #attfp_preds = get_model_preds(attentivefp, train_loader, device, is_attentivefp=True).flatten()
    #train_preds = np.stack([gcnn_preds, gnn_preds, attfp_preds], axis=1)
#
    ## Validation
    #gcnn_val_preds = get_model_preds(gcnn, val_loader, device).flatten()
    #gnn_val_preds = get_model_preds(gnn, val_loader, device).flatten()
    #attfp_val_preds = get_model_preds(attentivefp, val_loader, device, is_attentivefp=True).flatten()
    #val_preds = np.stack([gcnn_val_preds, gnn_val_preds, attfp_val_preds], axis=1)

    # Test
    gcnn_test_preds = get_model_preds(gcnn, test_loader, device).flatten()
    gnn_test_preds = get_model_preds(gnn, test_loader, device).flatten()
    attfp_test_preds = get_model_preds(attentivefp, test_loader, device, is_attentivefp=True).flatten()
    test_preds = np.stack([gcnn_test_preds, gnn_test_preds, attfp_test_preds], axis=1)

    #y_train = train_df[label_col].values.astype(np.float32)
    #y_val = val_df[label_col].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.float32)

    # ------------------------------------------------------------------------------
    # 3. Consensus by Average
    # ------------------------------------------------------------------------------
    # หาค่าเฉลี่ยของ y_pred (Probability) จากทั้ง 3 โมเดลในแนวแกนแถว (axis=1) ของทุกชุดข้อมูล
    #train_avg_prob = train_preds.mean(axis=1)
    #val_avg_prob   = val_preds.mean(axis=1)
    test_avg_prob  = test_preds.mean(axis=1)

    # หากค่าเฉลี่ย >= 0.5 ให้เป็นคลาส 1 (Active) นอกนั้นเป็น 0 (Inactive)
    #train_consensus_pred = (train_avg_prob >= 0.5).astype(int)
    #val_consensus_pred   = (val_avg_prob >= 0.5).astype(int)
    test_consensus_pred  = (test_avg_prob >= 0.5).astype(int)

    # พรินต์สมการ Consensus สรุปภาพรวม
    equation = "y_pred_final = 1 if (mean(GCNN + GMNN + AttentiveFP) >= 0.5) else 0"
    print("\nConsensus System (Simple Average):")
    print(equation)

    # บันทึกผลลัพธ์ลงไฟล์สำหรับแต่ละ seed
    output_txt_path = os.path.join(output_dir, 'consensus_summary.txt')
    with open(output_txt_path, 'a', encoding='utf-8') as f:
        f.write(f"Seed {seed}: {equation}\n")

    # ------------------------------------------------------------------------------
    # 4. Evaluate Consensus on Test Set
    # ------------------------------------------------------------------------------
    # ใช้ค่าความน่าจะเป็นเฉลี่ย (test_avg_prob) และคลาสที่ทำนายได้จาก Consensus (test_consensus_pred)
    probs = test_avg_prob
    y_pred = test_consensus_pred
    y_pred = (probs >= 0.5).astype(int)


    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    mcc = matthews_corrcoef(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"\nConsensus Test Performance: Acc={acc:.4f}, AUROC={auc:.4f}, AUPRC={auprc:.4f}, MCC={mcc:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Precision={precision:.4f}")

    # Final Test Performance
    print("\n📊 Final Test Set Evaluation")
    print(f"[{name},{acc:.4f},{auc:.4f},{auprc:.4f},{mcc:.4f}, {recall:.4f},{f1:.4f},{precision:.4f}]")

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
        results_df = pd.read_csv(result_file).reindex(columns=list(result_row))
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        results_df = pd.DataFrame(columns=list(result_row))

    # Append the new result and save back to the file
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, index=False)

    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")


if __name__ == "__main__":
    for seed in SEEDS:
        print(seed)
        main(seed)
