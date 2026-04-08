import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from rdkit import Chem
from torch_geometric.data import Data

# --- Atom and Bond Features ---
def atom_features(atom):
    return np.array([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        atom.GetTotalNumHs(),
        int(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.GetMass(),
    ], dtype=np.float32)

def mol_to_graph(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = np.array([atom_features(atom) for atom in mol.GetAtoms()])
    if label is not None:
        return x, label
    return x

# --- Load and Preprocess Data ---
def load_data(file_path, smiles_col, label_col):
    df = pd.read_csv(file_path)
    df[label_col] = df[label_col].map({"Active": 1, "Inactive": 0})
    df = df.dropna(subset=[smiles_col])
    return df

def extract_graph_features(df, smiles_col, label_col):
    features, labels = [], []
    for _, row in df.iterrows():
        graph, label = mol_to_graph(row[smiles_col], row[label_col])
        if graph is not None:
            features.append(graph.mean(axis=0))  # Use mean aggregation for graph features
            labels.append(label)
    return np.array(features), np.array(labels)

# --- Metrics ---
def evaluate(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    mcc = matthews_corrcoef(y_true, y_pred)
    return acc, auc, auprc, mcc

# --- Main ---
if __name__ == "__main__":
    name = "RF"
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

    # Extract graph features and labels
    X_train, y_train = extract_graph_features(train_df, smiles_col, label_col)
    X_val, y_val = extract_graph_features(val_df, smiles_col, label_col)
    X_test, y_test = extract_graph_features(test_df, smiles_col, label_col)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the test set
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]
    acc, auc, auprc, mcc = evaluate(y_test, test_pred, test_prob)
    print(f"\n📊 Test Set Evaluation")
    print(f"Test Performance | Acc: {acc:.4f} | AUROC: {auc:.4f} | AUPRC: {auprc:.4f} | MCC: {mcc:.4f}")

    # Y-randomization test
    y_randomization_results = []

    for i in range(10):
        print(f"\n🔄 Y-Randomization Iteration {i + 1}")
        
        # Shuffle labels in the training dataset
        y_train_random = np.random.permutation(y_train)

        # Train Random Forest model with shuffled labels
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_random)

        # Evaluate on the test set
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)[:, 1]
        acc, auc, auprc, mcc = evaluate(y_test, test_pred, test_prob)
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
    y_randomization_df.to_csv("y_randomization_results_rf.csv", index=False)

    print("\nY-randomization test completed. Results saved to 'y_randomization_results_rf.csv'.")