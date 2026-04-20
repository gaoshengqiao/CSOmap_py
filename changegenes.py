"""
In-silico gene change experiment.
Equivalent to MATLAB changegenes.m
Pure Python version - no .mat files.
"""
import os
import pickle
import numpy as np
from analyst import Analyst
from preprocess import preprocess
from reconstruct_3d import reconstruct_3d


def changegenes(cancertype, clusters, genes, newTPMs):
    """
    Change several genes in several clusters, set TPM to newTPMs,
    then run CSOmap and output an analyst object to .pkl file.
    """
    origindatapath = os.path.join('output', cancertype)
    originlabelpath = os.path.join('data', cancertype)

    a = Analyst(origindatapath, originlabelpath, os.path.join(origindatapath, 'result'), stat=False)
    cells_mask = np.zeros(len(a.labels), dtype=bool)
    for cluster in clusters:
        cells_mask |= (a.labels == (a.standards.index(cluster) + 1))
    allindex = np.arange(len(a.labels))
    cellsindex = allindex[cells_mask]

    newTPM_list = []
    for i, gene in enumerate(genes):
        gene_idx = a.genes.index(gene)
        oldTPM = a.TPM[gene_idx, :].copy()
        oldTPM[cellsindex] = newTPMs[i]
        newTPM_list.append(oldTPM)
    newTPM = np.vstack(newTPM_list)

    newcancertype = f"pseudo_{cancertype}_{clusters[0]}_change_{genes[0]}_to_{newTPMs[0]}"
    newdatapath = os.path.join('data', cancertype)
    newoutputpath = os.path.join('output', newcancertype)

    preprocess(newdatapath, newoutputpath, pseudo=True, genestochange=genes,
               TPMtochange=newTPM, newcells=[], newcellsTPM=[], cellstoknock=[])
    reconstruct_3d(newoutputpath, newoutputpath, 3, False, 50, 'tight')
    c = Analyst(newoutputpath, newdatapath, os.path.join(newoutputpath, 'result'), stat=True)

    out_dir = 'pseudo_changegenes_analysts'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, f'{newcancertype}_pseudo_workspace.pkl'), 'wb') as f:
        pickle.dump(c, f)

    c.affinitymatshow(normalize=0, fontsize=12, filename=f'{newcancertype}_affinitymat')
    c.writeresult3d(f'{newcancertype}_coordinate')
    c.countsshow(fontsize=8, filename=f'{newcancertype}_statistical_counts')
    c.writecounts(newcancertype)
    c.statisticsshow(fontsize=12, filename=f'{newcancertype}_statistical_results')
    c.reversestatisticsshow(fontsize=12, filename=f'{newcancertype}_reverse_statistical_results')
    c.writestatistics(f'{newcancertype}_connection')
    c.drawconclusion(0.05, f'{newcancertype}_conclusion')
    c.savegif(f'{newcancertype}_3dplot', f'{newcancertype}_3dplot')
    c.mainLR(f'{newcancertype}_mainLR')


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 5:
        ctype = sys.argv[1]
        clus = sys.argv[2].split(',')
        gns = sys.argv[3].split(',')
        ntps = [float(x) for x in sys.argv[4].split(',')]
        changegenes(ctype, clus, gns, ntps)
    else:
        print("Usage: python changegenes.py <cancertype> <clusters> <genes> <newTPMs>")
