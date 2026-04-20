# CSOmap Python (Python Version)

Python translation of the MATLAB package **CSOmap** from:
https://github.com/zhongguojie1998/CSOmap

Original paper:
"Reconstruction of cell spatial organization from single-cell RNA sequencing data based on ligand-receptor mediated self-assembly"

## Installation

This package requires Python >= 3.8 and the following dependencies:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels Pillow
```

Or install from the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Place your data in a folder under `data/<dataset_name>/`:

- **TPM.txt**: Tab-separated matrix. First column = gene names (NOT IDs). First row = cell names (with a leading tab `\t`). Values = TPM (NOT log-transformed).
- **label.txt**: Two columns, tab-separated. First column = cell names, second column = cell type labels.
  - If the first row is a header like `cells\tlabels`, it will be detected and skipped automatically.
- **LR_pairs.txt**: Three columns, tab-separated. ligand, receptor, interaction weight.

See the original repository for demo data examples.

### 2. Run Main Pipeline

```python
from runme import runme

runme('demo', condition='tight')
```

Or from command line:

```bash
python runme.py demo
```

Outputs will be saved in `output/demo/`.

### 3. Use Analyst Object

```python
from analyst import Analyst

a = Analyst('output/demo/', 'data/demo/', 'output/demo/result/', stat=True)
a.scatter3d(Title='demo 3d', filename='demo_3d')
a.writestatistics('statistics')
a.drawconclusion(0.05, 'conclusion')
```

### 4. In-silico Experiments

**Change gene expression:**
```python
from changegenes import changegenes
changegenes('demo', ['clusterA'], ['CD63'], [100.0])
```

**Knockout cells:**
```python
from knockoutcells import knockoutcells
knockoutcells('demo', ['clusterA'])
```

## File Structure

| File | Description |
|------|-------------|
| `preprocess.py` | Load raw data, filter LR pairs, save `data.pkl` |
| `reconstruct_3d.py` | Calculate affinity matrix and optimize 3D coordinates, save `workspace.pkl` |
| `myoptimize.py` | Core t-SNE-like gradient descent optimizer |
| `analyst.py` | Main analysis class (statistics, plotting, exports) |
| `runme.py` | One-line pipeline wrapper |
| `changegenes.py` | In-silico gene modification experiment |
| `knockoutcells.py` | In-silico cell knockout experiment |
| `draw_pictures/` | Plotting functions using matplotlib/seaborn |

## Differences from MATLAB Version

- **File format**: `.mat` → `.pkl` (Python pickle).
- **Statistics**: `mafdr` → `statsmodels.stats.multitest.multipletests` (Benjamini-Hochberg FDR).
- **PCA**: MATLAB `pca()` → `sklearn.decomposition.PCA`.
- **Plotting**: MATLAB `gramm` → `matplotlib` + `seaborn`.
- **Fast t-SNE**: Not implemented; use `condition='loose'` or `'tight'` instead.
- **Multi-core**: Vectorized NumPy replaces `parfor`.
- **Labels**: Output files use text labels directly; header rows in `label.txt` are auto-skipped.

## Notes

- The algorithm uses random initialization. Results may vary slightly across runs, but spatial patterns and statistical results should remain robust.
- For very large datasets (>30,000 cells), the code automatically downsamples to 10,000 cells during reconstruction, consistent with the MATLAB version.
