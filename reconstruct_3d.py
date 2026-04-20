"""
3D reconstruction for CSOmap.
Equivalent to MATLAB reconstruct_3d.m
Pure Python version - no .mat files.
"""
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from myoptimize import myoptimize


def calculate_affinity_mat(TPM, ligandindex, receptorindex, scores, denoise=0):
    """Calculate the affinity matrix from ligand and receptor TPM."""
    Atotake = ligandindex.copy()
    Btotake = receptorindex.copy()
    allscores = scores.copy()
    for i in range(len(ligandindex)):
        if ligandindex[i] != receptorindex[i]:
            Atotake = np.append(Atotake, receptorindex[i])
            Btotake = np.append(Btotake, ligandindex[i])
            allscores = np.append(allscores, scores[i])

    A = TPM[Atotake, :].T  # shape: (n_cells, n_pairs)
    B = TPM[Btotake, :]    # shape: (n_pairs, n_cells)

    if denoise == 0:
        affinitymat = (A * allscores) @ B.T
    else:
        affinitymat = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            row = np.zeros(B.shape[1])
            for j in range(B.shape[1]):
                row[j] = A[i, :] @ (B[:, j] * allscores)
            row[i] = 0
            sorted_row = np.sort(row)[::-1]
            threshold = sorted_row[min(denoise, len(sorted_row) - 1)]
            row[row < threshold] = 0
            affinitymat[i, :] = row
    return affinitymat


def calculate_affinity_mat_multi_cores(TPM, ligandindex, receptorindex, scores, denoise=0):
    """Calculate the affinity matrix (vectorized version)."""
    Atotake = ligandindex.copy()
    Btotake = receptorindex.copy()
    allscores = scores.copy()
    for i in range(len(ligandindex)):
        if ligandindex[i] != receptorindex[i]:
            Atotake = np.append(Atotake, receptorindex[i])
            Btotake = np.append(Btotake, ligandindex[i])
            allscores = np.append(allscores, scores[i])

    A = TPM[Atotake, :].T  # shape: (n_cells, n_pairs)
    B = TPM[Btotake, :]    # shape: (n_pairs, n_cells)

    if denoise == 0:
        affinitymat = (A * allscores) @ B.T
    else:
        affinitymat = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            row = np.zeros(B.shape[1])
            for j in range(B.shape[1]):
                row[j] = A[i, :] @ (B[:, j] * allscores)
            row[i] = 0
            sorted_row = np.sort(row)[::-1]
            threshold = sorted_row[min(denoise, len(sorted_row) - 1)]
            row[row < threshold] = 0
            affinitymat[i, :] = row
    return affinitymat


def denoising(originmat, k):
    """Denoise the affinity matrix, keep top k connections per cell."""
    if originmat.shape[0] <= k:
        return originmat
    result = originmat.copy()
    n = originmat.shape[0]
    for i in range(n):
        row = originmat[i, :]
        idx = np.argsort(row)[::-1]
        result[i, idx[k:]] = 0
    result = (result + result.T) / 2.0
    return result


def reconstruct_3d(datapath, outputpath, dim=3, use_single_core=False, denoise=50, condition='tobedetermined'):
    """
    Reconstruct 3D coordinates from preprocessed data.
    """
    # Import data
    with open(os.path.join(datapath, 'data.pkl'), 'rb') as f:
        data = pickle.load(f)

    TPM = data['TPM']
    cells = data['cells']
    genes = data['genes']
    ligandindex = data['ligandindex'].flatten().astype(int)
    receptorindex = data['receptorindex'].flatten().astype(int)
    scores = data['scores'].flatten()
    ligands = data['ligands']
    receptors = data['receptors']
    before = int(data.get('before', len(ligandindex)))
    after = int(data.get('after', len(ligandindex)))

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    else:
        print('Warning! Output path already exists, this program might change the files in it')

    if condition == 'tobedetermined':
        if TPM.shape[1] >= 9000:
            condition = 'loose'
            use_fast_tsne = False
            if TPM.shape[1] >= 30000:
                use_fast_tsne = True
        else:
            condition = 'tight'
            use_fast_tsne = False
    else:
        use_fast_tsne = False

    # Downsample if too many cells
    if TPM.shape[1] >= 30000:
        print(f"Down sampling from {TPM.shape[1]} to 10000")
        cellstotake = np.random.choice(TPM.shape[1], 10000, replace=False)
        cells = [cells[i] for i in cellstotake]
        TPM = TPM[:, cellstotake]

    # Calculate affinity matrix
    if use_single_core:
        affinitymat = calculate_affinity_mat(TPM, ligandindex, receptorindex, scores, denoise)
    else:
        affinitymat = calculate_affinity_mat_multi_cores(TPM, ligandindex, receptorindex, scores, denoise)

    # Optimize
    if use_fast_tsne:
        raise NotImplementedError("Fast t-SNE is not implemented in Python version. Please use condition='loose' or 'tight'.")
    else:
        result3d, process = myoptimize(affinitymat, 3, condition)

    pca = PCA(n_components=3)
    pca.fit(result3d)
    result3d = pca.transform(result3d)
    process_3d = process.copy()

    if dim != 3:
        result2d, process2d = myoptimize(affinitymat, dim, condition)
        pca2 = PCA(n_components=dim)
        pca2.fit(result2d)
        result2d = pca2.transform(result2d)
    else:
        result2d = None
        process2d = None

    # Save workspace as pickle
    workspace = {
        'TPM': TPM,
        'cells': cells,
        'genes': genes,
        'ligands': ligands,
        'receptors': receptors,
        'scores': scores,
        'ligandindex': ligandindex,
        'receptorindex': receptorindex,
        'affinitymat': affinitymat,
        'result3d': result3d,
        'process': process_3d,
        'dim': dim,
    }
    if result2d is not None:
        workspace['result2d'] = result2d
        workspace['process2d'] = process2d

    with open(os.path.join(outputpath, 'workspace.pkl'), 'wb') as f:
        pickle.dump(workspace, f)

    # Write information
    info_path = os.path.join(outputpath, 'information.txt')
    with open(info_path, 'w') as f:
        f.write(f'ligand-receptor pair number: {before}\n')
        f.write(f'ligand-receptor found in data set: {after}\n')
    print(f"3D reconstruction completed. Results saved to {outputpath}")
    return workspace


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 2:
        reconstruct_3d(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python reconstruct_3d.py <datapath> <outputpath>")
