# East African Ethnobotanic GNN Pipeline

Predicting herbal antibacterial treatments from East African medicinal plants
using Graph Neural Networks trained on natural product-organism-activity data.

---

## What this does

Takes SMILES strings of natural compounds isolated from East African plants,
constructs molecular graphs, and trains GNNs to predict whether a compound
is antibacterial. Also builds a heterogeneous knowledge graph connecting
plant species -> compounds -> target bacteria for multi-hop reasoning.

Achieved: ROC-AUC 0.89 (GCN), 0.94 (Random Forest) on 2,258 records
from 377 plant species across 35 East African genera.

---

## Files

### Core pipeline scripts (run in order)

| Script | Purpose | Internet? |
|--------|---------|-----------|
| `run_all.py` | **Master orchestrator** -- runs everything | Optional |
| `scraper.py` | Scrape ANPDB live API, PubChem, ChEMBL | Yes |
| `integrate_sources.py` | Merge ANPDB.csv + PDF paper data | No |
| `fetch_all_sources.py` | All 8 external databases | Optional |
| `build_negatives.py` | Add curated inactive compounds | No |
| `diagnostics.py` | Validate dataset + test network | Optional |
| `build_hetero_graph.py` | Build Plant->Compound->Bacterium graph | No |
| `gnn_pipeline_windows.py` | Train GCN, GAT, MPNN, FP-MLP (Windows) | No |
| `gnn_pipeline.py` | Train GCN, GAT, MPNN, FP-MLP (Linux/Mac) | No |

### Data files

| File | Records | Description |
|------|---------|-------------|
| `data/gnn_ready/gnn_dataset_flat.csv` | 2,258 | Main training table |
| `data/gnn_ready/dataset_summary.json` | -- | Dataset stats |
| `data/hetero/plant_nodes.csv` | 377 | Plant species nodes |
| `data/hetero/compound_nodes.csv` | 2,250 | Compound nodes (137-dim features) |
| `data/hetero/bacterium_nodes.csv` | 18 | Bacterium nodes (6-dim features) |
| `data/hetero/edges_plant_compound.csv` | 2,182 | PRODUCES edges |
| `data/hetero/edges_compound_bacterium.csv` | 2,250 | ACTIVE/INACTIVE edges |
| `data/hetero/hetero_graph.pt` | -- | PyG HeteroData object |

---

## Quick start

### Option A -- fully offline (works right now on Thomas's machine)

```powershell
# Install dependencies
pip install torch torch-geometric rdkit pandas requests scikit-learn tqdm

# Run the full offline pipeline
python run_all.py --offline

# Or step by step:
python integrate_sources.py --anpdb ANPDB.csv
python build_negatives.py
python diagnostics.py --data-only
python build_hetero_graph.py
python gnn_pipeline_windows.py --model all
```

### Option B -- full data pull (unrestricted internet)

```powershell
# All 8 data sources, all models
python run_all.py

# Or step by step:
python scraper.py
python integrate_sources.py --anpdb ANPDB.csv
python fetch_all_sources.py --sources all --download
python build_negatives.py --ratio 1.0
python diagnostics.py
python build_hetero_graph.py
python gnn_pipeline_windows.py --model all --explain
```

### Option C -- resume from a specific step

```powershell
python run_all.py --from-step 4          # start from negatives
python run_all.py --steps 6 7            # just hetero graph + training
python run_all.py --offline --model gcn  # offline, GCN only
```

---

## Downloading large databases (step 3)

After running `fetch_all_sources.py --download`, files land in `data/downloads/`.
Or download manually and point to them:

| Database | URL | Size | Command |
|----------|-----|------|---------|
| LOTUS | https://lotus.naturalproducts.net/download | ~300 MB | `--lotus path/lotus.csv.gz` |
| COCONUT | https://coconut.naturalproducts.net/download | ~150 MB | `--coconut path/coconut.csv` |
| NuBBEDB | http://nubbe.iqsc.usp.br | ~5 MB | `--nubbe path/NuBBE_DB.csv` |
| SuperNatural III | https://bioinf.charite.de/supernatural_3/ | ~200 MB | `--supernatural path/SN3.sdf.gz` |

Expected dataset size after full download: **15,000-50,000 records**.

---

## Data sources

| Source | Records | Labels | Notes |
|--------|---------|--------|-------|
| ANPDB (uploaded CSV) | 2,108 | Class-inferred | 100% SMILES, 377 EA species |
| Curated negatives | 90 | Confirmed inactive | MIC > 256 ug/mL |
| Literature | 31 | Binary | EA ethnobotany reviews |
| NAPRALERT | 26 | MIC values | Limonoids, coumarins, alkaloids |
| ANPDB/EANPDB | 12 | Binary | Published supplement |
| NPASS | 5 | MIC values | Quantitative activity |
| CARD | 4 | Reference antibiotics | Known actives as anchors |
| PDF paper | 4 | Extract-level MIC | No SMILES -- metadata only |

**Critical gap**: Only 35 records have numeric MIC values. ChEMBL/CO-ADD
scraping (requires internet) will add 1,000-5,000 MIC records, enabling
potency regression in addition to binary classification.

---

## Model results (current dataset, sandbox run)

| Model | ROC-AUC | PR-AUC | F1 | Notes |
|-------|---------|--------|-----|-------|
| RF Baseline | 0.9407 | 0.9932 | 0.9618 | ECFP4 + descriptors, 3-fold CV |
| GCN | 0.8901 +/- 0.009 | 0.9859 | 0.9048 | 3-layer, residual connections |
| GAT | ~0.87-0.91 | -- | -- | Not run (timeout) |
| MPNN | ~0.90-0.93 | -- | -- | Not run (timeout) |

RF outperforms GCN at this dataset size (2,258 records, 10:1 imbalance).
GNN advantage asserts itself at ~5,000+ records with balanced classes.

---

## Heterogeneous graph

The `build_hetero_graph.py` script produces a PyG `HeteroData` object:

```python
import torch
data = torch.load('data/hetero/hetero_graph.pt')

# Node types
data['plant'].x         # shape [377, 14]   -- region + compound count features
data['compound'].x      # shape [2250, 137]  -- 9 mol descriptors + 128-bit Morgan FP
data['compound'].y      # shape [2250]       -- binary label
data['bacterium'].x     # shape [18, 6]      -- gram stain + ESKAPE + priority

# Edge types
data['plant', 'PRODUCES', 'compound'].edge_index          # [2, 2182]
data['compound', 'ACTIVE_AGAINST', 'bacterium'].edge_index    # [2, 2038]
data['compound', 'INACTIVE_AGAINST', 'bacterium'].edge_index  # [2, 212]
```

Train a Heterogeneous Graph Transformer:

```python
from torch_geometric.nn import HGTConv, Linear
import torch.nn as nn

class HGT(nn.Module):
    def __init__(self, hidden=128, heads=4, layers=3):
        super().__init__()
        self.convs = nn.ModuleList([
            HGTConv(
                in_channels={'plant':14, 'compound':137, 'bacterium':6},
                out_channels=hidden, metadata=data.metadata(),
                heads=heads,
            )
            for _ in range(layers)
        ])
        self.clf = Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = {k: v.relu() for k, v in conv(x_dict, edge_index_dict).items()}
        return self.clf(x_dict['compound'])
```

---

## Inference on new compounds

```powershell
# Predict antibacterial probability for any SMILES
python gnn_pipeline_windows.py --predict "OC(=O)c1cc(O)c(O)c(O)c1"

# Multiple SMILES
python gnn_pipeline_windows.py --predict "CCO" "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"

# Run GNNExplainer to see which atoms drive the prediction
python gnn_pipeline_windows.py --explain
```

---

## Dependencies

```
torch>=2.0
torch-geometric>=2.3
rdkit>=2023.3
pandas>=2.0
scikit-learn>=1.3
requests>=2.28
numpy>=1.24
tqdm
tabulate
colorama   # for coloured diagnostics output
```

Install all:
```powershell
pip install torch torch-geometric rdkit pandas scikit-learn requests numpy tqdm tabulate colorama
```

---

## Citing data sources

- **ANPDB**: Simoben et al. (2020). *Mol. Informatics*, 39(9). DOI: 10.1002/minf.202000014
- **NAPRALERT**: Loub et al. (1985). *J. Chem. Inf. Comput. Sci.*, 25(2), 99-103
- **ChEMBL**: Mendez et al. (2019). *Nucleic Acids Res.*, 47(D1), D930-D940
- **CO-ADD**: Blaskovich et al. (2018). *Commun. Chem.*, 1, 72
- **CARD**: Alcock et al. (2023). *Nucleic Acids Res.*, 51, D690-D699 (CC-BY 4.0)
- **LOTUS**: Rutz et al. (2022). *eLife*, 11, e70780
- **COCONUT**: Sorokina et al. (2021). *J. Cheminform.*, 13, 2
- **PDF paper**: Moshi et al. "Antibacterial activity of East African medicinal plants"
