import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
import time


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
    if mol is None:
        return None

    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index, edge_attr = [], []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]  # Undirected
        feat = bond_features(bond)
        edge_attr += [feat, feat]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)

    return data


# --- Load and Preprocess Data ---
def load_data(file_path, smiles_col, label_col):
    df = pd.read_csv(file_path)
    df[label_col] = df[label_col].map({"Active": 1, "Inactive": 0})
    df = df.dropna(subset=[smiles_col])
    return df


def convert_to_graphs(df, smiles_col, label_col):
    df['graph'] = df.apply(lambda row: mol_to_graph(row[smiles_col], row[label_col]), axis=1)
    return df['graph'].dropna().tolist()


# --- GNN Model ---
class GNNClassifier(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64):
        super(GNNClassifier, self).__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim * hidden_dim)
        )
        self.nnconv = NNConv(
            in_channels=node_dim,
            out_channels=hidden_dim,
            nn=self.edge_net,
            aggr='mean'
        )
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.nnconv(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(1)


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

    return acc, auc, auprc, mcc


# --- Main ---
if __name__ == "__main__":
    name = "GNN"
    print(f"Model: {name}")
    print("📊 Loading and preprocessing data...\n")
    
    # Start timing the entire process
    start_time = time.time()
    
    # Load and preprocess data
    file_path = "mapk1_darlene.csv"
    smiles_col = "SMILES"
    label_col = "Label"

    df = load_data(file_path, smiles_col, label_col)
    print(f"Number of valid SMILES: {len(df)}")

    # Load train, validation, and test sets
    train_df = pd.read_csv("train.csv")
    val_df = pd.read_csv("val.csv")
    test_df = pd.read_csv("test.csv")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Convert validation and test sets to graphs (unchanged across iterations)
    val_graphs = convert_to_graphs(val_df, smiles_col, label_col)
    test_graphs = convert_to_graphs(test_df, smiles_col, label_col)

    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    # Y-randomization test
    y_randomization_results = []

    for i in range(10):
        print(f"\n🔄 Y-Randomization Iteration {i + 1}")
        
        # Shuffle labels only in the training dataset
        train_df[label_col] = np.random.permutation(train_df[label_col].values)

        # Convert the permuted training set to graphs
        train_graphs = convert_to_graphs(train_df, smiles_col, label_col)
        train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        node_dim = train_graphs[0].x.shape[1]
        edge_dim = train_graphs[0].edge_attr.shape[1]

        model = GNNClassifier(node_dim, edge_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        pos_weight = torch.tensor([len(train_df[train_df[label_col] == 0]) /
                                   len(train_df[train_df[label_col] == 1])], dtype=torch.float).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

        acc, auc, auprc, mcc = evaluate(model, test_loader, device)
        print(f"Iteration {i + 1} | Test Acc: {acc:.4f} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f}")
        
        y_randomization_results.append({
            "Model": name,
            "Iteration": i + 1,
            "Accuracy": round(acc, 4),
            "AUROC": round(auc, 4),
            "AUPRC": round(auprc, 4),
            "MCC": round(mcc, 4)
        })

    # Save Y-randomization results
    y_randomization_df = pd.DataFrame(y_randomization_results)
    y_randomization_df.to_csv("y_randomization_results_gnn.csv", index=False)

    print("\nY-randomization test completed. Results saved to 'y_randomization_results_gnn.csv'.")