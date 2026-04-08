import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt

# === 0. สร้างโฟลเดอร์เก็บรูป === 
output_dir = "scaffold_grouping"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Folder '{output_dir}' ถูกสร้างแล้ว (หรือมีอยู่แล้ว)")

# === 1. Draw Scaffold ===
def draw_scaffold(smiles, name, output_dir=output_dir, img_size=(400, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"⚠️ SMILES ไม่ถูกต้อง: {smiles}")
        return

    AllChem.Compute2DCoords(mol)

    # === วาด SVG แบบไม่มี highlight ===
    drawer = rdMolDraw2D.MolDraw2DSVG(*img_size)
    drawer.drawOptions().clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    with open(os.path.join(output_dir, f"{name}.svg"), "w") as f:
        f.write(svg)

    print(f"✅ บันทึก: {name}.svg")

def scaffold_extract(df):
    scaffold_counts = {}
    for smiles in df['canonical_smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
            scaffold_counts[scaffold_smiles] = scaffold_counts.get(scaffold_smiles, 0) + 1

    # === 4. เรียง scaffold จากมากไปน้อย และเลือก Top 10 ===
    sorted_scaffolds = sorted(scaffold_counts.items(), key=lambda x: x[1], reverse=True)
    top_scaffolds = sorted_scaffolds[:10]

    print("🔝 Top 10 Active scaffolds (SMILES, Count)")
    for scaffold, count in top_scaffolds:
        print(f"{scaffold}\t{count}")
    return top_scaffolds

def bar_plot_scaffolds(scaffold_labels, counts, color = "tomato",figsizes=(10, 5), ylabel= "SMILES of Active Scaffold"):
    plt.figure(figsize=figsizes)
    bars = plt.barh(scaffold_labels, counts, color=color, edgecolor='black')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(int(width)), va='center')

    plt.xlabel("Number of Molecules", fontsize=12, fontweight='bold', style='italic')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold', style='italic')
    plt.gca().invert_yaxis()  # ให้ Top1 อยู่ด้านบน
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Top10_"+ylabel+".svg"), format='svg')
    plt.close()
    print("🎯 เสร็จสิ้น: วาด Top 10 "+ylabel+"เรียบร้อย")

def main():
    df = pd.read_csv("mapk1_darlene.csv", index_col=0)
    # Load Active
    df_sel = df[df['Class'] == 'Active'].copy()
    print(f"✅ เลือกเฉพาะ Class == 'Active' แล้ว จำนวนโมเลกุล: {len(df_sel)}")
    top_scaffolds = scaffold_extract(df_sel)
    for i, (scaffold_smiles, count) in enumerate(top_scaffolds, start=1):
        name = f"Top{i}_Count{count}"
        draw_scaffold(scaffold_smiles, name)
    scaffold_labels = [smiles for smiles, _ in top_scaffolds]  # ใช้ SMILES จริง
    counts = [count for _, count in top_scaffolds]
    bar_plot_scaffolds(scaffold_labels, counts, color = "tomato", figsizes=(10, 5), ylabel= "SMILES of Active Scaffold")
    # Load Inactive
    df_sel = df[df['Class'] == 'Inactive'].copy()
    print(f"✅ เลือกเฉพาะ Class == 'Inactive' แล้ว จำนวนโมเลกุล: {len(df_sel)}")
    top_scaffolds = scaffold_extract(df_sel)
    for i, (scaffold_smiles, count) in enumerate(top_scaffolds, start=1):
        name = f"Top{i}_Count{count}"
        draw_scaffold(scaffold_smiles, name)
    scaffold_labels = [smiles for smiles, _ in top_scaffolds]  # ใช้ SMILES จริง
    counts = [count for _, count in top_scaffolds]
    bar_plot_scaffolds(scaffold_labels, counts, color="deepskyblue", figsizes=(10, 5), ylabel="SMILES of Inactive Scaffold")


if __name__ == "__main__":
    main()
    