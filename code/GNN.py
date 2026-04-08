import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
import time  # Import the time module

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


def is_valid_molecule(smiles):
    if not isinstance(smiles, str) or '.' in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


# --- Load and Preprocess Data ---
def load_data(file_path, smiles_col, label_col):
    df = pd.read_csv(file_path)
    df[label_col] = df[label_col].map({"Active": 1, "Inactive": 0})
    df = df[df[smiles_col].apply(is_valid_molecule)].reset_index(drop=True)
    df = df.dropna(subset=[smiles_col])
    return df


def split_data(df, smiles_col, label_col):
    train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df[label_col], random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df[label_col], random_state=42)
    # save train, test, and validation
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    return train_df, val_df, test_df


def convert_to_graphs(df, smiles_col, label_col):
    df['graph'] = df.apply(lambda row: mol_to_graph(row[smiles_col], row[label_col]), axis=1)
    return df['graph'].dropna().tolist()


# --- Model ---
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



# --- SHAP Explanation Helper ---
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def shap_forward(x_numpy, model):
    """
    Forward function for SHAP that treats each atom as an isolated node (no edges)
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_numpy, dtype=torch.float)
        edge_index = torch.tensor([], dtype=torch.long).view(2, 0)  # no edges
        batch = torch.zeros(x.size(0), dtype=torch.long)  # all nodes in one graph
        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = model(data)
        return out.detach().cpu().numpy()

def run_shap_explainer(model, test_graphs, feature_names=None, name="GCNN"):
    """
    Runs SHAP KernelExplainer on each atom individually in the first test graph.
    """
    if feature_names is None:
        feature_names = [
            "AtomicNum", "Degree", "FormalCharge", "Chirality",
            "NumH", "Hybridization", "Aromaticity", "AtomicMass"
        ]

    sample_data = test_graphs[0]
    x = sample_data.x.cpu().numpy()
    shap_values_all = []

    for i in range(x.shape[0]):
        single_atom_x = x[i:i+1]
        background = single_atom_x  # one-atom background

        explainer = shap.KernelExplainer(lambda x_: shap_forward(x_, model), background)
        shap_values = explainer.shap_values(single_atom_x)
        shap_values_all.append(shap_values)

        print(f"Atom {i} SHAP values: {shap_values}")

    # Flatten and plot summary across all atoms
    combined_x = np.vstack([x[i:i+1] for i in range(x.shape[0])])
    combined_shap = np.vstack([sv[0] for sv in shap_values_all])
    # Plot
    plt.figure(figsize=(3, 4))
    shap.summary_plot(combined_shap, combined_x, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"shap_summary_{name}.pdf", dpi=500, bbox_inches='tight')
    plt.show()

    return shap_values_all

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

    train_df, val_df, test_df = split_data(df, smiles_col, label_col)

    train_graphs = convert_to_graphs(train_df, smiles_col, label_col)
    val_graphs = convert_to_graphs(val_df, smiles_col, label_col)
    test_graphs = convert_to_graphs(test_df, smiles_col, label_col)

    print(f"📊 Train: {len(train_graphs)} | Val: {len(val_graphs)} | Test: {len(test_graphs)}")

    # Loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32)
    test_loader = DataLoader(test_graphs, batch_size=32)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]

    model = GNNClassifier(node_dim, edge_dim).to(device)
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
        acc, auc, auprc, mcc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f}")

    # After training, save the model weights
    model_save_path = "gnn_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel weights saved to '{model_save_path}'")

    # Final Test Performance
    print("\n📊 Final Test Set Evaluation")
    acc, auc, auprc, mcc = evaluate(model, test_loader, device)
    # Print results in the specified format
    print(f"[{name},{acc:.4f},{auc:.4f},{auprc:.4f},{mcc:.4f}]")

    # Append results to result.csv
    result_row = {
        "Model": name,
        "Accuracy": round(acc, 3),
        "AUROC": round(auc, 3),
        "AUPRC": round(auprc, 3),
        "MCC": round(mcc, 3),
        "Time (s)": round(time.time() - start_time, 2)  # Total time for the run
    }
    result_file = "result.csv"

    try:
        # Load existing results if the file exists
        results_df = pd.read_csv(result_file)
    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        results_df = pd.DataFrame(columns=["Model", "Accuracy", "AUROC", "AUPRC", "MCC"])

    # Append the new result and save back to the file
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, index=False)
    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")