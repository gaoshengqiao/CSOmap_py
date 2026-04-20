"""
Data preprocessing for CSOmap.
Equivalent to MATLAB preprocess.m
Pure Python version - no .mat files.
"""
import os
import pickle
import numpy as np
import pandas as pd


def preprocess(rawdatapath, processed_datapath, pseudo=False, LRpairstoadd=None,
               genestochange=None, TPMtochange=None, newcells=None, newcellsTPM=None,
               cellstoknock=None):
    """
    Import raw data and process them into a pickle file.
    """
    if LRpairstoadd is None:
        LRpairstoadd = []
    if genestochange is None or TPMtochange is None:
        genestochange = []
        TPMtochange = []
    if newcells is None or newcellsTPM is None:
        newcells = []
        newcellsTPM = []
    if cellstoknock is None:
        cellstoknock = []

    # Load ligand-receptor pairs
    lr_path = os.path.join(rawdatapath, 'LR_pairs.txt')
    lr_df = pd.read_csv(lr_path, sep='\t', header=None, names=['ligand', 'receptor', 'score'], dtype=str)
    lr_df['score'] = lr_df['score'].astype(float)
    ligands = lr_df['ligand'].tolist()
    receptors = lr_df['receptor'].tolist()
    scores = lr_df['score'].values

    # Load TPM data
    tpm_path = os.path.join(rawdatapath, 'TPM.txt')
    tpm_df = pd.read_csv(tpm_path, sep='\t', index_col=0)
    TPM = tpm_df.values.astype(float)
    # Force gene and cell names to strings to avoid int/float mismatch
    genes = [str(g).strip() for g in tpm_df.index.tolist()]
    cells = [str(c).strip() for c in tpm_df.columns.tolist()]

    if not os.path.exists(processed_datapath):
        os.makedirs(processed_datapath)
    else:
        print('Warning! directory already exists, this program might change the files in it')

    # Pseudo mode: add new LR pairs
    if pseudo and len(LRpairstoadd) > 0:
        for item in LRpairstoadd:
            ligands.append(str(item[0]))
            receptors.append(str(item[1]))
            scores = np.append(scores, float(item[2]))

    # Pseudo mode: change gene expression
    if pseudo and len(genestochange) > 0:
        for i, gene in enumerate(genestochange):
            gene = str(gene).strip()
            if gene in genes:
                idx = genes.index(gene)
                TPM[idx, :] = TPMtochange[i, :]
            else:
                genes.append(gene)
                TPM = np.vstack([TPM, TPMtochange[i, :]])

    # Pseudo mode: add new cells
    if pseudo and len(newcells) > 0:
        newcells = [str(c).strip() for c in newcells]
        cells = cells + list(newcells)
        TPM = np.hstack([TPM, newcellsTPM])

    # Pseudo mode: knock out cells
    if pseudo and len(cellstoknock) > 0:
        mask = np.ones(len(cells), dtype=bool)
        mask[cellstoknock] = False
        cells = [cells[i] for i in range(len(cells)) if mask[i]]
        TPM = TPM[:, mask]

    # Calculate ligand and receptor indexes in TPM matrix
    ligandindex = np.array([genes.index(l) if l in genes else -1 for l in ligands])
    receptorindex = np.array([genes.index(r) if r in genes else -1 for r in receptors])
    found = (ligandindex != -1) & (receptorindex != -1)

    before = len(ligandindex)
    ligandindex = ligandindex[found]
    receptorindex = receptorindex[found]
    ligands = [ligands[i] for i in range(len(ligands)) if found[i]]
    receptors = [receptors[i] for i in range(len(receptors)) if found[i]]
    scores = scores[found]
    after = len(ligandindex)

    # Save processed data as pickle
    data_dict = {
        'TPM': TPM,
        'genes': genes,
        'cells': cells,
        'ligands': ligands,
        'receptors': receptors,
        'scores': scores,
        'ligandindex': ligandindex,
        'receptorindex': receptorindex,
        'before': before,
        'after': after
    }
    save_path = os.path.join(processed_datapath, 'data.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Preprocessed data saved to {save_path}")
    print(f"LR pairs before filtering: {before}, after filtering: {after}")
    return data_dict


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        preprocess(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python preprocess.py <rawdatapath> <processed_datapath>")
