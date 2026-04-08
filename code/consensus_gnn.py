import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool, NNConv, AttentiveFP
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time  # Import the time module
import matplotlib.pyplot as plt
from rdkit import Chem

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

    return acc, auc, auprc, mcc


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

# --- Meta Model ---
class MetaDense(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.fc(x).squeeze(1)

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

# --- Plotting ---
def plot_confusion_matrix_with_percentages(y_true, y_pred, class_names, name="StackGNN"):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_percent[i] = cm[i] / row_sums[i] * 100

    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", ax=ax, values_format="d", colorbar=False)

    # Add raw numbers and percentages to the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black", fontsize=10)
            ax.text(j, i + 0.2, f"({cm_percent[i, j]:.1f}%)", ha="center", va="center", color="black", fontsize=9)

    plt.title(f"Confusion Matrix_{name}", fontsize=12, fontweight="bold", style="italic")
    plt.xlabel("Predicted values", fontsize=12, fontweight="bold", style="italic")
    plt.ylabel("True values", fontsize=12, fontweight="bold", style="italic")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_with_{name}.pdf", dpi=500)
    plt.close()

# --- Main ---
if __name__ == "__main__":
    name = "ConsensusGNN"
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
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]

    # --- Load pretrained base models ---
    gcnn = GCNNClassifier(node_dim).to(device)
    gcnn.load_state_dict(torch.load("gcnn_model.pth", map_location=device))
    gcnn.eval()

    gnn = GNNClassifier(node_dim, edge_dim).to(device)
    gnn.load_state_dict(torch.load("gnn_model.pth", map_location=device))
    gnn.eval()

    attentivefp = AttentiveFP(
        in_channels=node_dim,
        hidden_channels=64,
        out_channels=1,
        edge_dim=edge_dim,
        num_layers=3,
        num_timesteps=2
    ).to(device)
    attentivefp.load_state_dict(torch.load("attentivefp_model.pth", map_location=device))
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

    y_train = train_df["Label"].values.astype(np.float32)
    y_val = val_df["Label"].values.astype(np.float32)
    y_test = test_df["Label"].values.astype(np.float32)

    # 3. Consensus (average) ensemble
    probs = np.mean([gcnn_test_preds, gnn_test_preds, attfp_test_preds], axis=0)
    y_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"\nConsensus Ensemble Test Performance: Acc={acc:.4f}, AUROC={auc:.4f}, AUPRC={auprc:.4f}, MCC={mcc:.4f}")

    # Save the consensus probabilities if needed
    np.save("consensus_probs.npy", probs)

    # Final Test Performance
    print("\n📊 Final Test Set Evaluation")
    print(f"[ConsensusGNN,{acc:.4f},{auc:.4f},{auprc:.4f},{mcc:.4f}]")

    # Append results to result.csv
    result_row = {
        "Model": "ConsensusGNN",
        "Accuracy": round(acc, 3),
        "AUROC": round(auc, 3),
        "AUPRC": round(auprc, 3),
        "MCC": round(mcc, 3),
        "Time (s)": round(time.time() - start_time, 2)  # Total time for the run
    }
    result_file = "result.csv"

    try:
        results_df = pd.read_csv(result_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=["Model", "Accuracy", "AUROC", "AUPRC", "MCC", "Time (s)"])

    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, index=False)

    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")

    # Plot Confusion Matrix for test set
    plot_confusion_matrix_with_percentages(
        y_true=y_test,
        y_pred=y_pred,
        class_names=["Inactive", "Active"],
        name="ConsensusGNN"
    )