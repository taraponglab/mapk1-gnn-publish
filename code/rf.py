import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef
from rdkit import Chem
import torch
from torch_geometric.data import Data
import os
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


def convert_to_graphs(df, smiles_col, label_col):
    df['graph'] = df.apply(lambda row: mol_to_graph(row[smiles_col], row[label_col]), axis=1)
    return df['graph'].dropna().tolist()


def load_datasets(smiles_col, label_col):
    print("\U0001F4CA Loading train, validation, and test datasets...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_graphs = convert_to_graphs(train_df, smiles_col, label_col)
    test_graphs = convert_to_graphs(test_df, smiles_col, label_col)

    return train_graphs, test_graphs


def evaluate(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, auc, auprc, mcc


if __name__ == "__main__":
    name = "RF-Graph"
    smiles_col = "SMILES"
    label_col = "Label"
    
    # Start timing the entire process
    start_time = time.time()

    train_graphs, test_graphs = load_datasets(smiles_col, label_col)

    # Extract features and labels from graphs
    X_train = np.array([graph.x.mean(dim=0).numpy() for graph in train_graphs])  # Use mean aggregation
    y_train = np.array([graph.y.item() for graph in train_graphs])
    X_test = np.array([graph.x.mean(dim=0).numpy() for graph in test_graphs])  # Use mean aggregation
    y_test = np.array([graph.y.item() for graph in test_graphs])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=np.random.randint(0, 100000))
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]
    acc, auc, auprc, mcc = evaluate(y_train, train_pred, train_prob)
    print(f"\n\U0001F4CA Training Set Evaluation")
    print(f"\nTrain Performance | Acc: {acc:.4f} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f}")

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    acc, auc, auprc, mcc = evaluate(y_test, test_pred, test_prob)
    print("\n\U0001F4CA Final Test Set Evaluation")
    print(f"[{name},{acc:.4f},{auc:.4f},{auprc:.4f},{mcc:.4f}]")

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
        results_df = pd.read_csv(result_file)
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=["Model", "Accuracy", "AUROC", "AUPRC", "MCC"])
    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, index=False)

    # Save the Random Forest model and scaler
    model_save_path = os.path.join("models", f"{name.lower()}_model.joblib")
    os.makedirs("models", exist_ok=True)  # Ensure the "models" directory exists
    joblib.dump((model, scaler), model_save_path)
    print(f"\n✅ Model and scaler saved as '{model_save_path}'")

    # Print total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")





