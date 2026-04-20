"""
In-silico cell knock-out experiment.
Equivalent to MATLAB knockoutcells.m
Pure Python version - no .mat files.
"""
import os
import pickle
import numpy as np
from analyst import Analyst
from preprocess import preprocess
from reconstruct_3d import reconstruct_3d


def knockoutcells(cancertype, clusters):
    """
    Knock out a cluster in dataset, then run CSOmap and output an analyst object to .pkl file.
    """
    a = Analyst(os.path.join('output', cancertype),
                os.path.join('data', cancertype),
                os.path.join('output', cancertype, 'result'), stat=False)
    cells_mask = np.zeros(len(a.labels), dtype=bool)
    for cluster in clusters:
        cells_mask |= (a.labels == (a.standards.index(cluster) + 1))
    allindex = np.arange(len(a.labels))
    cellsindex = allindex[cells_mask]

    newcancertype = f"pseudo_{cancertype}_knockout_{clusters[0]}"
    newdatapath = os.path.join('data', cancertype)
    newoutputpath = os.path.join('output', newcancertype)

    preprocess(newdatapath, newoutputpath, pseudo=True, LRpairstoadd=[],
               genestochange=[], TPMtochange=[], newcells=[], newcellsTPM=[],
               cellstoknock=cellsindex.tolist())
    reconstruct_3d(newoutputpath, newoutputpath, 3, False, 50, 'tight')
    c = Analyst(newoutputpath, newdatapath, os.path.join(newoutputpath, 'result'), stat=True)

    out_dir = 'pseudo_knockoutcells_analysts'
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
    if len(sys.argv) >= 3:
        ctype = sys.argv[1]
        clus = sys.argv[2].split(',')
        knockoutcells(ctype, clus)
    else:
        print("Usage: python knockoutcells.py <cancertype> <clusters>")
