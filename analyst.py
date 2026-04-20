"""
Analyst class for CSOmap.
Equivalent to MATLAB analyst.m
Stores all data and provides methods for statistical analysis and plotting.
Pure Python version - no .mat files.
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata, hypergeom, norm
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

warnings.filterwarnings('ignore')


class Analyst:
    def __init__(self, workspace_path, labelpath, finaloutputpath, stat=True):
        """
        Load data from workspace.pkl and generate an Analyst object.
        If stat is True, perform statistical analysis.
        workspace_path can be a directory (containing workspace.pkl) or the path to workspace.pkl itself.
        """
        if os.path.isdir(workspace_path):
            pkl_path = os.path.join(workspace_path, 'workspace.pkl')
        else:
            pkl_path = workspace_path

        with open(pkl_path, 'rb') as f:
            ws = pickle.load(f)

        cells = ws['cells']
        # Ensure cells are strings
        cells = [str(c).strip() for c in cells]

        labels, standards = self.identify_label(cells, labelpath)

        self.TPM = ws['TPM']
        self.genes = [str(g).strip() for g in ws['genes']]
        self.affinitymat = ws['affinitymat']
        self.cells = cells
        self.labels = labels
        self.standards = standards
        self.ligands = [str(l).strip() for l in ws['ligands']]
        self.receptors = [str(r).strip() for r in ws['receptors']]
        self.scores = ws['scores'].flatten()
        self.ligandindex = ws['ligandindex'].flatten().astype(int)
        self.receptorindex = ws['receptorindex'].flatten().astype(int)
        self.result3d = ws['result3d']
        self.process = ws['process']
        self.outputpath = finaloutputpath
        self.connection = None
        self.reverseconnection = None
        self.neighbor = None
        self.degree = None
        self.counts = None
        self.clustercounts = None
        self.result2d = ws.get('result2d', np.array([]))
        self.dim = int(ws.get('dim', 3))

        if not os.path.exists(finaloutputpath):
            os.makedirs(finaloutputpath)
        else:
            print('Warning! directory already exists, this program might change the files in it')

        if stat:
            self.getconnection(3)

    # ------------------------------------------------------------------
    # Main statistical functions
    # ------------------------------------------------------------------
    def getconnection(self, k, dim=3, useaffinitymat=0, method='hpgdistri', porq=False):
        """
        Identify connections between cells.
        k: median number of connections.
        dim: dimension (default 3).
        useaffinitymat: if not 0, use affinity matrix directly.
        method: 'hpgdistri' or 'permutation'.
        porq: False for q-value, True for p-value.
        """
        if dim != 3 and self.result2d.size == 0:
            tsneresult = self.result3d
        elif dim != 3:
            tsneresult = self.result2d
        else:
            tsneresult = self.result3d

        n = tsneresult.shape[0]
        labelsize = int(self.labels.max())
        if labelsize <= 1:
            print('Error: there is only one kind of label in your dataset, unable to do statistical analysis.')
            return
        if n <= k:
            print('obj not changed, because k is smaller than cell number')
            return

        if useaffinitymat == 0:
            distancemat = self.getdistancemat(tsneresult)
            distancemat[np.arange(n), np.arange(n)] = np.inf
            cutoffs = np.zeros(n)
            for i in range(n):
                row = distancemat[i, :]
                B = np.sort(row)
                cutoffs[i] = B[k]
            cutoff = np.median(cutoffs)
            row_idx, col_idx = np.where(distancemat <= cutoff)
        else:
            distancemat = useaffinitymat.copy()
            distancemat[np.arange(n), np.arange(n)] = 0
            cutoffs = np.zeros(n)
            for i in range(n):
                row = distancemat[i, :]
                B = np.sort(row)[::-1]
                cutoffs[i] = B[k]
            cutoff = np.median(cutoffs)
            row_idx, col_idx = np.where(distancemat >= cutoff)

        # Initialize counts
        objcounts = [[np.zeros((0, 2), dtype=int) for _ in range(labelsize)] for _ in range(labelsize)]
        objclustercounts = np.bincount(self.labels, minlength=labelsize + 1)[1:]
        objneighbor = [[np.zeros((0, 1), dtype=int) for _ in range(labelsize)] for _ in range(n)]
        realconnection = np.zeros((labelsize, labelsize))

        for ri, ci in zip(row_idx, col_idx):
            realconnection[self.labels[ri] - 1, self.labels[ci] - 1] += 1
            objcounts[self.labels[ri] - 1][self.labels[ci] - 1] = np.vstack([
                objcounts[self.labels[ri] - 1][self.labels[ci] - 1],
                [ri + 1, ci + 1]  # 1-based to match MATLAB convention
            ])
            objneighbor[ri][self.labels[ci] - 1] = np.vstack([
                objneighbor[ri][self.labels[ci] - 1],
                [ci + 1]  # 1-based
            ])

        realconnection = realconnection - 0.5 * np.diag(np.diag(realconnection))

        if method == 'permutation':
            realconnection_flat = realconnection.flatten()
            randomtotal = np.zeros((len(realconnection_flat), 1000))
            for j in range(1000):
                randomconnection = np.zeros((labelsize, labelsize))
                randomlabels = self.labels[np.random.permutation(len(self.labels))]
                for ri, ci in zip(row_idx, col_idx):
                    randomconnection[randomlabels[ri] - 1, randomlabels[ci] - 1] += 1
                randomconnection = randomconnection - 0.5 * np.diag(np.diag(randomconnection))
                randomtotal[:, j] = randomconnection.flatten()

            p_value = np.ones(len(realconnection_flat))
            reverse_p_value = np.zeros(len(realconnection_flat))
            for i in range(len(realconnection_flat)):
                muhat, sigmahat = norm.fit(randomtotal[i, :])
                if realconnection_flat[i] != 0:
                    p_value[i] = 1 - norm.cdf(realconnection_flat[i], muhat, sigmahat)
                    reverse_p_value[i] = norm.cdf(realconnection_flat[i], muhat, sigmahat)

            _, q_value, _, _ = multipletests(p_value, method='fdr_bh')
            _, reverse_q_value, _, _ = multipletests(reverse_p_value, method='fdr_bh')
            q_value = q_value.reshape((labelsize, labelsize))
            reverse_q_value = reverse_q_value.reshape((labelsize, labelsize))
        elif method == 'hpgdistri':
            pop = n * (n - 1) / 2
            samplenumber = int(len(row_idx) / 2)
            p_value = np.ones((labelsize, labelsize))
            reverse_p_value = np.zeros((labelsize, labelsize))
            for i in range(labelsize):
                for j in range(i, labelsize):
                    if i == j:
                        propertynumber = objclustercounts[i] * (objclustercounts[i] - 1) / 2
                    else:
                        propertynumber = objclustercounts[i] * objclustercounts[j]
                    eventnumber = int(realconnection[i, j])
                    if eventnumber == 0:
                        p_value[i, j] = 1
                        reverse_p_value[i, j] = 0
                    else:
                        p_value[i, j] = 1 - hypergeom.cdf(eventnumber - 1, int(pop), int(propertynumber), samplenumber)
                        reverse_p_value[i, j] = hypergeom.cdf(eventnumber, int(pop), int(propertynumber), samplenumber)
                    p_value[j, i] = p_value[i, j]
                    reverse_p_value[j, i] = reverse_p_value[i, j]

            _, q_value, _, _ = multipletests(p_value.flatten(), method='fdr_bh')
            _, reverse_q_value, _, _ = multipletests(reverse_p_value.flatten(), method='fdr_bh')
            q_value = q_value.reshape((labelsize, labelsize))
            reverse_q_value = reverse_q_value.reshape((labelsize, labelsize))
        else:
            raise ValueError(f"Unknown method: {method}")

        if porq:
            self.connection = p_value
            self.reverseconnection = reverse_p_value
        else:
            self.connection = q_value
            self.reverseconnection = reverse_q_value

        self.neighbor = objneighbor
        # Correct diagonal counts (each connection counted twice)
        for i in range(labelsize):
            count = objcounts[i][i]
            if count.shape[0] > 0:
                corrected = []
                seen = set()
                for c in count:
                    key = f"{min(c)}-{max(c)}"
                    if key not in seen:
                        seen.add(key)
                        corrected.append(c)
                objcounts[i][i] = np.array(corrected, dtype=int)

        self.counts = objcounts
        self.clustercounts = objclustercounts
        self.degree = np.zeros((n, labelsize), dtype=int)
        for i in range(n):
            for j in range(labelsize):
                self.degree[i, j] = objneighbor[i][j].shape[0]

    # ------------------------------------------------------------------
    # Affinity matrix functions
    # ------------------------------------------------------------------
    def affinitymathistogram(self, filename=None, Title='histogram of affinity'):
        fig, ax = plt.subplots()
        ax.hist(self.affinitymat.flatten(), bins=50)
        ax.set_title(Title)
        if filename:
            fig.savefig(os.path.join(self.outputpath, f'{filename}.jpg'), dpi=150)
        return fig

    def affinitymatshow(self, normalize=False, fontsize=12, filename=None):
        sorted_idx = np.argsort(self.labels)
        Paffinitymat = self.affinitymat[:, sorted_idx]
        Paffinitymat = Paffinitymat[sorted_idx, :]
        if normalize:
            normalizefactor = np.diag(np.sqrt(1.0 / (Paffinitymat.mean(axis=1) + 1e-10)))
            Paffinitymat = normalizefactor @ Paffinitymat @ normalizefactor

        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(Paffinitymat, aspect='auto', cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        for i in range(len(self.standards)):
            positions = np.where(self.labels[sorted_idx] == i + 1)[0]
            if len(positions) == 0:
                continue
            position1 = positions[0] - 0.5
            if i != len(self.standards) - 1:
                positions2 = np.where(self.labels[sorted_idx] == i + 2)[0]
                if len(positions2) > 0:
                    position2 = positions2[0] - 0.5
                    position = (position1 + position2) / 2
                else:
                    position = (position1 + Paffinitymat.shape[0] + 0.5) / 2
            else:
                position = (position1 + Paffinitymat.shape[0] + 0.5) / 2
            ax.axhline(y=position1, color='red', linewidth=0.5)
            ax.axvline(x=position1, color='red', linewidth=0.5)
            ax.text(-0.02, position / Paffinitymat.shape[0],
                    f"{i + 1}:{self.standards[i]}-",
                    color='red', fontsize=fontsize, ha='right', va='center',
                    transform=ax.transAxes)
            ax.text(position / Paffinitymat.shape[1], -0.02,
                    f"-{i + 1}:{self.standards[i]}",
                    color='red', fontsize=fontsize, ha='left', va='top',
                    rotation=30, transform=ax.transAxes)
        if filename:
            fig.savefig(os.path.join(self.outputpath, f'{filename}.jpg'), dpi=150)
        return fig

    def writeaffinitymat(self, filename):
        path = os.path.join(self.outputpath, f'{filename}.txt')
        with open(path, 'w') as f:
            f.write('affinity\t')
            f.write('\t'.join(self.cells) + '\n')
            for i in range(self.affinitymat.shape[0]):
                f.write(self.cells[i] + '\t')
                f.write('\t'.join([f'{x:.4f}' for x in self.affinitymat[i, :]]) + '\n')

    def writeTPM(self, filename):
        path = f'{filename}.txt'
        with open(path, 'w') as f:
            f.write('Gene\t')
            f.write('\t'.join(self.cells) + '\n')
            for i in range(self.TPM.shape[0]):
                f.write(self.genes[i] + '\t')
                f.write('\t'.join([f'{x:.4f}' for x in self.TPM[i, :]]) + '\n')

    def writelabels(self, filename):
        path = f'{filename}.txt'
        with open(path, 'w') as f:
            f.write('Cell\tLabel\n')
            for i in range(len(self.labels)):
                f.write(f'{self.cells[i]}\t{self.standards[self.labels[i] - 1]}\n')

    def writeresult3d(self, filename):
        path = os.path.join(self.outputpath, f'{filename}.txt')
        with open(path, 'w') as f:
            f.write('ID\tx\ty\tz\tlabel\n')
            for i in range(len(self.cells)):
                label_str = self.standards[self.labels[i] - 1]
                f.write(f'{self.cells[i]}\t{self.result3d[i, 0]:f}\t{self.result3d[i, 1]:f}\t{self.result3d[i, 2]:f}\t{label_str}\n')

    # ------------------------------------------------------------------
    # Process show
    # ------------------------------------------------------------------
    def processshow(self, filename=None, show=None):
        if show is None:
            show = filename is None
        iters = self.process.shape[1] // self.result3d.shape[1]
        dim = 3
        images = []
        for iter in range(iters):
            current_coord = self.process[:, iter * dim:iter * dim + 3]
            fig = self._tdplot(current_coord, self.labels, self.standards, filename or 'process show')
            if filename:
                fig.canvas.draw()
                img = np.asarray(fig.canvas.buffer_rgba())
                img = img[:, :, :3]
                images.append(img)
                plt.close(fig)
            elif not show:
                plt.close(fig)
        if filename and len(images) > 0:
            from PIL import Image
            gif_path = os.path.join(self.outputpath, f'{filename}.gif')
            images_pil = [Image.fromarray(im) for im in images]
            images_pil[0].save(gif_path, save_all=True, append_images=images_pil[1:], duration=20, loop=0)

    def _tdplot(self, result3d, labels, standards, Title):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if len(labels) > 0:
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1, labels.max()))
            for i in range(1, labels.max() + 1):
                toplot = result3d[labels == i, :]
                if len(toplot) == 0:
                    continue
                ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                           c=[colors[i - 1, :3]], s=20, label=standards[i - 1])
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.set_title(Title)
        return fig

    # ------------------------------------------------------------------
    # Counts functions
    # ------------------------------------------------------------------
    def countsshow(self, fontsize=12, filename=None):
        if self.counts is None or len(self.counts) == 0:
            print('counts not found in object, now it will run "getconnection" first')
            self.getconnection(3)
        labelsize = len(self.standards)
        alphadata = np.ones((labelsize, labelsize))
        for i in range(labelsize):
            for j in range(labelsize):
                if j > i:
                    alphadata[i, j] = 0
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.connection, aspect='auto', cmap='viridis', alpha=alphadata)
        plt.colorbar(im, ax=ax, orientation='horizontal')
        ax.axis('off')
        for i in range(labelsize):
            position1 = i - 0.5
            position = i
            ax.axhline(y=position1, xmin=0.05, xmax=(position + 0.5) / labelsize, color='white', linewidth=0.5)
            ax.axvline(x=(position + 0.5) / labelsize, ymin=position1 / labelsize, ymax=1, color='white', linewidth=0.5)
            ax.text(0.02, (position + 0.5) / labelsize, f"{i + 1}:{self.standards[i]}-",
                    color='red', fontsize=fontsize, ha='right', va='center', transform=ax.transAxes)
            ax.text((position + 0.5) / labelsize, 0.02, f"-{i + 1}:{self.standards[i]}",
                    color='red', fontsize=fontsize, ha='left', va='bottom', rotation=20, transform=ax.transAxes)
            for j in range(i + 1):
                count = self.counts[i][j]
                countnumber = count.shape[0]
                if countnumber:
                    if j != i:
                        countfromA = len(np.unique(count[:, 0]))
                        countfromB = len(np.unique(count[:, 1]))
                    else:
                        countfromA = len(np.unique(np.concatenate([count[:, 0], count[:, 1]])))
                        countfromB = countfromA
                else:
                    countfromA = 0
                    countfromB = 0
                ax.text(j, i, f"{countnumber}\n{countfromA}/{countfromB}\n{self.clustercounts[i]}/{self.clustercounts[j]}",
                        color='white', fontsize=fontsize, ha='center', va='center')
        if filename:
            fig.savefig(os.path.join(self.outputpath, f'{filename}.jpg'), dpi=150)
        return fig

    def writecounts(self, filename):
        path1 = os.path.join(self.outputpath, f'{filename}_connections.txt')
        path2 = os.path.join(self.outputpath, f'{filename}_cellcounts.txt')
        path3 = os.path.join(self.outputpath, f'{filename}_totalcells.txt')
        with open(path1, 'w') as f:
            f.write('number\t' + '\t'.join(self.standards) + '\n')
            for i in range(len(self.counts)):
                f.write(self.standards[i] + '\t')
                f.write('\t'.join([str(self.counts[i][j].shape[0]) for j in range(len(self.counts[i]))]) + '\n')
        with open(path2, 'w') as f:
            f.write('clusterA/clusterB\t' + '\t'.join(self.standards) + '\n')
            for i in range(len(self.counts)):
                f.write(self.standards[i] + '\t')
                vals = []
                for j in range(len(self.counts[i])):
                    if self.counts[i][j].shape[0]:
                        if i == j:
                            vals.append(str(len(np.unique(np.concatenate([self.counts[i][j][:, 0], self.counts[i][j][:, 1]])))))
                        else:
                            vals.append(str(len(np.unique(self.counts[i][j][:, 0]))))
                    else:
                        vals.append('')
                f.write('\t'.join(vals) + '\n')
        with open(path3, 'w') as f:
            f.write('clusterA/clusterB\t' + '\t'.join(self.standards) + '\n')
            for i in range(len(self.counts)):
                f.write(self.standards[i] + '\t')
                f.write('\t'.join([str(self.clustercounts[i])] * len(self.counts[i])) + '\n')

    # ------------------------------------------------------------------
    # Degree functions
    # ------------------------------------------------------------------
    def writedegree(self, filename):
        path = os.path.join(self.outputpath, f'{filename}.txt')
        with open(path, 'w') as f:
            f.write('degree\t' + '\t'.join(self.standards) + '\n')
            for i in range(self.degree.shape[0]):
                f.write(self.standards[i] + '\t')
                f.write('\t'.join([str(self.degree[i, j]) for j in range(self.connection.shape[1])]) + '\n')

    # ------------------------------------------------------------------
    # Differential genes
    # ------------------------------------------------------------------
    def differential_genes(self, clusterA, clusterB, pcutoff=0.05, fcutoff=1.0,
                           testtype='ttest', draw=False, filename=None):
        from scipy import stats
        if isinstance(clusterA, str):
            clusterA = self.standards.index(clusterA) + 1
        if isinstance(clusterB, str):
            clusterB = self.standards.index(clusterB) + 1
        allindexes = np.arange(len(self.labels)) + 1
        allAcells = allindexes[self.labels == clusterA]
        if clusterA != clusterB:
            if self.counts[clusterA - 1][clusterB - 1].shape[0] > 0:
                ABcells = np.unique(self.counts[clusterA - 1][clusterB - 1][:, 0])
            else:
                ABcells = np.array([], dtype=int)
        else:
            if self.counts[clusterA - 1][clusterB - 1].shape[0] > 0:
                ABcells = np.unique(np.concatenate([
                    self.counts[clusterA - 1][clusterB - 1][:, 0],
                    self.counts[clusterA - 1][clusterB - 1][:, 1]
                ]))
            else:
                ABcells = np.array([], dtype=int)
        ABcellsTPM = self.TPM[:, ABcells - 1]
        AnotBcells = allAcells[~np.isin(allAcells, ABcells)]
        AnotBcellsTPM = self.TPM[:, AnotBcells - 1]

        if ABcellsTPM.shape[1] == 0 or AnotBcellsTPM.shape[1] == 0:
            print("Warning: one group is empty, cannot perform differential analysis.")
            return [], [], ABcellsTPM, AnotBcellsTPM

        if testtype == 'ttest':
            p = np.array([stats.ttest_ind(np.log2(ABcellsTPM[i, :] + 1),
                                          np.log2(AnotBcellsTPM[i, :] + 1)).pvalue
                          for i in range(ABcellsTPM.shape[0])])
        else:
            p = np.array([stats.ranksums(np.log2(ABcellsTPM[i, :] + 1),
                                         np.log2(AnotBcellsTPM[i, :] + 1)).pvalue
                          for i in range(ABcellsTPM.shape[0])])

        _, p, _, _ = multipletests(p, method='fdr_bh')
        fold_change = ABcellsTPM.mean(axis=1) / (AnotBcellsTPM.mean(axis=1) + 1e-10)

        if fcutoff > 0:
            mask = (p < pcutoff) & (fold_change > fcutoff)
        elif fcutoff < 0:
            mask = (p < pcutoff) & (fold_change < abs(fcutoff))
        else:
            mask = p < pcutoff
        genes = [self.genes[i] for i in range(len(self.genes)) if mask[i]]
        genes_foldchange = fold_change[mask]
        return genes, genes_foldchange, ABcellsTPM, AnotBcellsTPM

    # ------------------------------------------------------------------
    # Statistics show
    # ------------------------------------------------------------------
    def statisticsshow(self, fontsize=12, filename=None):
        labelsize = len(self.standards)
        alphadata = np.ones((labelsize, labelsize))
        for i in range(labelsize):
            for j in range(labelsize):
                if j > i:
                    alphadata[i, j] = 0
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.connection, aspect='auto', cmap='viridis', alpha=alphadata)
        plt.colorbar(im, ax=ax, orientation='horizontal')
        ax.axis('off')
        for i in range(labelsize):
            position1 = i - 0.5
            position = i
            ax.axhline(y=position1, xmin=0.05, xmax=(position + 0.5) / labelsize, color='white', linewidth=0.5)
            ax.axvline(x=(position + 0.5) / labelsize, ymin=position1 / labelsize, ymax=1, color='white', linewidth=0.5)
            ax.text(0.02, (position + 0.5) / labelsize, f"{i + 1}:{self.standards[i]}-",
                    color='red', fontsize=fontsize, ha='right', va='center', transform=ax.transAxes)
            ax.text((position + 0.5) / labelsize, 0.02, f"-{i + 1}:{self.standards[i]}",
                    color='red', fontsize=fontsize, ha='left', va='bottom', rotation=20, transform=ax.transAxes)
            for j in range(i + 1):
                ax.text(j, i, f"{self.connection[i, j]:.4f}",
                        color='white', fontsize=fontsize, ha='center', va='center')
        if filename:
            fig.savefig(os.path.join(self.outputpath, f'{filename}.jpg'), dpi=150)
        return fig

    def reversestatisticsshow(self, fontsize=12, filename=None):
        labelsize = len(self.standards)
        alphadata = np.ones((labelsize, labelsize))
        for i in range(labelsize):
            for j in range(labelsize):
                if j > i:
                    alphadata[i, j] = 0
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.reverseconnection, aspect='auto', cmap='viridis', alpha=alphadata)
        plt.colorbar(im, ax=ax, orientation='horizontal')
        ax.axis('off')
        for i in range(labelsize):
            position1 = i - 0.5
            position = i
            ax.axhline(y=position1, xmin=0.05, xmax=(position + 0.5) / labelsize, color='white', linewidth=0.5)
            ax.axvline(x=(position + 0.5) / labelsize, ymin=position1 / labelsize, ymax=1, color='white', linewidth=0.5)
            ax.text(0.02, (position + 0.5) / labelsize, f"{i + 1}:{self.standards[i]}-",
                    color='red', fontsize=fontsize, ha='right', va='center', transform=ax.transAxes)
            ax.text((position + 0.5) / labelsize, 0.02, f"-{i + 1}:{self.standards[i]}",
                    color='red', fontsize=fontsize, ha='left', va='bottom', rotation=20, transform=ax.transAxes)
            for j in range(i + 1):
                ax.text(j, i, f"{self.reverseconnection[i, j]:.4f}",
                        color='white', fontsize=fontsize, ha='center', va='center')
        if filename:
            fig.savefig(os.path.join(self.outputpath, f'{filename}.jpg'), dpi=150)
        return fig

    def writestatistics(self, filename):
        path = os.path.join(self.outputpath, f'{filename}.txt')
        with open(path, 'w') as f:
            f.write('q-value\t' + '\t'.join(self.standards) + '\n')
            for i in range(self.connection.shape[0]):
                f.write(self.standards[i] + '\t')
                f.write('\t'.join([f'{x:.4f}' for x in self.connection[i, :]]) + '\n')

    def drawconclusion(self, pcutoff, filename):
        conclusion = []
        for i in range(len(self.standards)):
            for j in range(i + 1):
                if self.connection[i, j] <= pcutoff:
                    conclusion.append(f"{self.standards[i]}---{self.standards[j]}")
        conclusion = sorted(conclusion)
        path = os.path.join(self.outputpath, f'{filename}.txt')
        with open(path, 'w') as f:
            for item in conclusion:
                f.write(item + '\n')
        return conclusion

    # ------------------------------------------------------------------
    # Coordinate presentation functions
    # ------------------------------------------------------------------
    def scatter2d(self, Title='scatter 2d', filename=None):
        if self.result2d.size == 0:
            pca = PCA(n_components=2)
            Presult = pca.fit_transform(self.result3d)
        else:
            Presult = self.result2d
        fig, ax = plt.subplots(figsize=(12, 10))
        if len(self.labels) > 0:
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1, self.labels.max()))
            for i in range(1, self.labels.max() + 1):
                toplot = Presult[self.labels == i, :]
                if len(toplot) == 0:
                    continue
                ax.scatter(toplot[:, 0], toplot[:, 1], c=[colors[i - 1, :3]], s=20, label=self.standards[i - 1])
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.set_title(Title)
        if filename:
            fig.savefig(os.path.join(self.outputpath, f'{filename}.jpg'), dpi=150)
        return fig

    def scatter3d(self, Title='scatter 3d', filename=None, iter=None):
        if iter is not None:
            # iter in MATLAB is 1-based column index
            coordinates = self.process[:, iter - 1:iter + 2]
        else:
            coordinates = self.result3d
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        if len(self.labels) > 0:
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, self.labels.max()))
            for i in range(1, self.labels.max() + 1):
                toplot = coordinates[self.labels == i, :]
                if len(toplot) == 0:
                    continue
                ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                           c=[colors[i - 1, :3]], s=50, label=self.standards[i - 1])
            ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.set_title(Title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 60)
        ax.set_zlim(-60, 60)
        plt.tight_layout()
        if filename:
            fig.savefig(os.path.join(self.outputpath, f'{filename}.pdf'), dpi=150)
        return fig

    def getsections(self, Axis='z', Title=None, filename=None):
        if Title is None:
            Title = f'{Axis} view'
        fig = self.scatter3d()
        ax = fig.axes[0]
        for i in range(-50, 31, 20):
            if Axis == 'x':
                ax.set_xlim(i, i + 20)
                ax.view_init(elev=0, azim=90)
                ax.set_title(f"{Title} range: [{i}, {i + 20}]")
            elif Axis == 'y':
                ax.set_ylim(i, i + 20)
                ax.view_init(elev=90, azim=0)
                ax.set_title(f"{Title} range: [{i}, {i + 20}]")
            elif Axis == 'z':
                ax.set_zlim(i, i + 20)
                ax.view_init(elev=0, azim=0)
                ax.set_title(f"{Title} range: [{i}, {i + 20}]")
            if filename:
                fig.savefig(os.path.join(self.outputpath,
                                         f"{filename}{Title} range[{i},{i + 20}].pdf"), dpi=150)
        if filename:
            plt.close(fig)
        return fig

    def scattercluster(self, cluster, Title=None, filename=None):
        if Title is None:
            Title = f'scatter of {cluster}'
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        i = self.standards.index(cluster) + 1
        if len(self.labels) > 0:
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1, self.labels.max()))
            toplot = self.result3d[self.labels == i, :]
            ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                       c=[colors[i - 1, :3]], s=20, label=self.standards[i - 1])
            ax.legend()
        ax.set_title(Title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if filename:
            fig.savefig(os.path.join(self.outputpath, f'{filename}.pdf'), dpi=150)
        return fig

    def savegif(self, Title, filename):
        fig = self.scatter3d(Title)
        ax = fig.axes[0]
        rotate_dir = os.path.join(self.outputpath, 'rotate')
        if not os.path.exists(rotate_dir):
            os.makedirs(rotate_dir)
        images = []
        for i in range(36):
            ax.view_init(elev=30, azim=10 * i)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())
            img = img[:, :, :3]
            images.append(img)
            fig.savefig(os.path.join(rotate_dir, f"{filename}_{(i + 1) * 10}.pdf"), dpi=150)
        plt.close(fig)
        if len(images) > 0:
            from PIL import Image
            gif_path = os.path.join(self.outputpath, f'{filename}.gif')
            images_pil = [Image.fromarray(im) for im in images]
            images_pil[0].save(gif_path, save_all=True, append_images=images_pil[1:], duration=250, loop=0)

    # ------------------------------------------------------------------
    # Gene expression
    # ------------------------------------------------------------------
    def expressionshow(self, gene, filename=None, Title=None):
        if Title is None:
            Title = f'Expression of {gene}'
        gene_idx = self.genes.index(gene)
        geneTPM = np.log2(self.TPM[gene_idx, :] + 1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(self.result3d[:, 0], self.result3d[:, 1], self.result3d[:, 2],
                        c=geneTPM, cmap='viridis', s=20)
        ax.set_title(Title)
        plt.colorbar(sc, ax=ax)
        if filename:
            fig.savefig(os.path.join(self.outputpath, filename), dpi=150)
        return fig

    # ------------------------------------------------------------------
    # Main contributed Ligand and Receptor
    # ------------------------------------------------------------------
    def mainLR(self, filename):
        path = os.path.join(self.outputpath, f'{filename}.txt')
        if self.counts is None or len(self.counts) == 0:
            print('counts not found in object, now it will run "getconnection" first')
            self.getconnection(3)
        Atotake = self.ligandindex.copy()
        Btotake = self.receptorindex.copy()
        LRpairs = list(zip(self.ligands, self.receptors))
        allscores = self.scores.copy()
        for i in range(len(self.ligandindex)):
            if self.ligandindex[i] != self.receptorindex[i]:
                Atotake = np.append(Atotake, self.receptorindex[i])
                Btotake = np.append(Btotake, self.ligandindex[i])
                LRpairs.append((self.receptors[i], self.ligands[i]))
                allscores = np.append(allscores, self.scores[i])

        with open(path, 'w') as f:
            for i in range(len(self.counts)):
                for j in range(i + 1):
                    if i == j:
                        cis_count = self.counts[i][j]
                        trans_count = cis_count[:, [1, 0]]
                        count = np.vstack([cis_count, trans_count])
                    else:
                        count = self.counts[i][j]
                    allconclusion = {}
                    if count.shape[0] > 0:
                        f.write(f"{self.standards[i]}---{self.standards[j]}\tq-value: {self.connection[i, j]}\n\n")
                        f.write("Ligand---Receptor\tcontribution\n")
                        for k in range(count.shape[0]):
                            A = self.TPM[Atotake, count[k, 0] - 1]
                            B = self.TPM[Btotake, count[k, 1] - 1]
                            affinity = np.sum(A * allscores * B)
                            if affinity == 0:
                                continue
                            contributes = A * allscores * B / affinity
                            top10_idx = np.argsort(contributes)[::-1][:10]
                            for l_idx in top10_idx:
                                key = f"{LRpairs[l_idx][0]}---{LRpairs[l_idx][1]}"
                                if key in allconclusion:
                                    allconclusion[key] += contributes[l_idx]
                                else:
                                    allconclusion[key] = contributes[l_idx]
                        sorted_items = sorted(allconclusion.items(), key=lambda x: x[1], reverse=True)
                        countnumber = count.shape[0]
                        for key, val in sorted_items:
                            f.write(f"{key}\t{val / countnumber:.6f}\n")
                        f.write("\n")
                    else:
                        f.write(f"{self.standards[i]}---{self.standards[j]}\tq-value: {self.connection[i, j]}\n\n")

    # ------------------------------------------------------------------
    # Change labels
    # ------------------------------------------------------------------
    def setnewlabels(self, newlabelpath, restat=True):
        newlabels, newstandards = self.identify_label(self.cells, newlabelpath)
        self.labels = newlabels
        self.standards = newstandards
        if restat:
            self.getconnection(3)
        else:
            self.connection = None
            self.reverseconnection = None
            self.counts = None
            self.clustercounts = None

    # ------------------------------------------------------------------
    # Cluster DP (Density Peak)
    # ------------------------------------------------------------------
    def cluster_dp(self, k=3, mds=False, type='cutoff', show=False):
        xx = self.getplaindistance(self.result3d)
        dist = self.getdistancemat(self.result3d)
        ND = int(xx[:, 1].max())
        print(f'median number of neighbours (hard coded): {k:5.6f}')
        dist[np.arange(ND), np.arange(ND)] = np.inf
        cutoffs = np.zeros(ND)
        for i in range(ND):
            row = dist[i, :]
            B = np.sort(row)
            cutoffs[i] = B[k]
        dc = np.median(cutoffs)
        dist[np.arange(ND), np.arange(ND)] = 0
        print(f'Computing Rho with gaussian kernel of radius: {dc:12.6f}')

        rho = np.zeros(ND)
        if type == 'gaussian':
            for i in range(ND - 1):
                for j in range(i + 1, ND):
                    v = np.exp(-(dist[i, j] / dc) ** 2)
                    rho[i] += v
                    rho[j] += v
        elif type == 'cutoff':
            for i in range(ND - 1):
                for j in range(i + 1, ND):
                    if dist[i, j] < dc:
                        rho[i] += 1
                        rho[j] += 1
        else:
            raise ValueError(f"Unknown type: {type}")

        ordrho = np.argsort(rho)[::-1]
        delta = np.zeros(ND)
        nneigh = np.zeros(ND, dtype=int)
        delta[ordrho[0]] = -1
        nneigh[ordrho[0]] = -1
        for ii in range(1, ND):
            delta[ordrho[ii]] = np.inf
            for jj in range(ii):
                if dist[ordrho[ii], ordrho[jj]] < delta[ordrho[ii]]:
                    delta[ordrho[ii]] = dist[ordrho[ii], ordrho[jj]]
                    nneigh[ordrho[ii]] = ordrho[jj]
        delta[ordrho[0]] = delta.max()

        rhomin = rho.mean() + 2 * rho.std()
        deltamin = delta.mean() + 2 * delta.std()
        NCLUST = 0
        cl = -np.ones(ND, dtype=int)
        icl = np.zeros(ND, dtype=int)
        for i in range(ND):
            if rho[i] > rhomin and delta[i] > deltamin:
                NCLUST += 1
                cl[i] = NCLUST
                icl[NCLUST - 1] = i
        print(f'NUMBER OF CLUSTERS: {NCLUST}')
        print('Performing assignation')

        for i in range(ND):
            if cl[ordrho[i]] == -1:
                cl[ordrho[i]] = cl[nneigh[ordrho[i]]]

        halo = cl.copy()
        if NCLUST > 1:
            bord_rho = np.zeros(NCLUST)
            for i in range(ND - 1):
                for j in range(i + 1, ND):
                    if cl[i] != cl[j] and dist[i, j] <= dc:
                        rho_aver = (rho[i] + rho[j]) / 2.0
                        if rho_aver > bord_rho[cl[i] - 1]:
                            bord_rho[cl[i] - 1] = rho_aver
                        if rho_aver > bord_rho[cl[j] - 1]:
                            bord_rho[cl[j] - 1] = rho_aver
            for i in range(ND):
                if rho[i] < bord_rho[cl[i] - 1]:
                    halo[i] = -1

        # Calculate distribution
        distribution = {}
        for i in range(1, NCLUST + 1):
            members = np.where(halo == i)[0]
            labels_in_cluster = self.labels[members]
            distribution[i] = {}
            for lab in np.unique(labels_in_cluster):
                distribution[i][self.standards[lab - 1]] = int(np.sum(labels_in_cluster == lab))

        return cl, halo, distribution

    # ------------------------------------------------------------------
    # Spatial non-randomly distributed genes
    # ------------------------------------------------------------------
    def spatial_nonrandom(self, gene, allrandomcells=None, dim=3, expressioncutoff=0,
                          numhist=None):
        n = len(self.cells)
        if allrandomcells is None:
            allrandomcells = np.zeros((n, 1000), dtype=int)
            for i in range(1000):
                allrandomcells[:, i] = np.random.permutation(n)
        if numhist is None:
            numhist = self.result3d.shape[0] // 10
            if numhist < 2:
                numhist = 2

        if dim == 3:
            coordinates = self.result3d
        elif dim == 2 and self.result2d.size > 0:
            coordinates = self.result2d
        else:
            coordinates = self.result3d

        mindist = 0
        maxdist = np.sqrt(np.sum((coordinates.max(axis=0) - coordinates.min(axis=0)) ** 2))
        step = (maxdist - mindist) / numhist
        histrange = np.arange(mindist + step / 2, maxdist, step)

        gene_idx = self.genes.index(gene)
        indexestotake = self.TPM[gene_idx, :] > expressioncutoff
        realcells = coordinates[indexestotake, :]
        if realcells.shape[0] < 2:
            return 1.0, 1.0, 0.5, 0.5, 1.0, 0.0
        real_dists = pdist(realcells)
        realdist, _ = np.histogram(real_dists, bins=histrange)
        realdist = realdist / (realcells.shape[0] * (realcells.shape[0] - 1) / 2)
        realmeandist = real_dists.mean() if len(real_dists) > 0 else 0

        randomdist = np.zeros((1000, len(realdist)))
        randommeandist = np.zeros(1000)
        for i in range(1000):
            randomcoordinates = coordinates[allrandomcells[:, i], :]
            randomcells = randomcoordinates[indexestotake, :]
            if randomcells.shape[0] < 2:
                continue
            rd = pdist(randomcells)
            rd_hist, _ = np.histogram(rd, bins=histrange)
            randomdist[i, :] = rd_hist / (realcells.shape[0] * (realcells.shape[0] - 1) / 2)
            randommeandist[i] = rd.mean() if len(rd) > 0 else 0

        meandist = randomdist.mean(axis=0)
        randombackground = np.sum(np.abs(randomdist - meandist), axis=1)
        realstat = np.sum(np.abs(realdist - meandist))

        muhat, sigmahat = norm.fit(randombackground)
        mu2, sig2 = norm.fit(randommeandist)
        p1 = 1 - norm.cdf(realstat, muhat, sigmahat)
        p2 = np.sum(randombackground >= realstat) / 1000.0
        p3 = 1 - norm.cdf(realmeandist, mu2, sig2)
        p4 = norm.cdf(realmeandist, mu2, sig2)
        p5 = np.sum(randommeandist >= realmeandist) / 1000.0
        p6 = np.sum(randommeandist <= realmeandist) / 1000.0
        print(f"{gene} {p1} {p2} {p3} {p4} {p5} {p6}")
        return p1, p2, p3, p4, p5, p6

    def all_spatial_nonrandom(self):
        n = len(self.cells)
        allrandomcells = np.zeros((n, 1000), dtype=int)
        for i in range(1000):
            allrandomcells[:, i] = np.random.permutation(n)
        num = len(self.genes)
        allgenes = self.genes
        p1 = np.zeros(num)
        p2 = np.zeros(num)
        p3 = np.zeros(num)
        p4 = np.zeros(num)
        p5 = np.zeros(num)
        p6 = np.zeros(num)
        for i in range(num):
            try:
                p_1, p_2, p_3, p_4, p_5, p_6 = self.spatial_nonrandom(allgenes[i], allrandomcells)
                p1[i] = p_1
                p2[i] = p_2
                p3[i] = p_3
                p4[i] = p_4
                p5[i] = p_5
                p6[i] = p_6
            except Exception as e:
                print(f'error in {self.genes[i]}: {e}')
        print('end of this data set')
        return p1, p2, p3, p4, p5, p6, allrandomcells, allgenes

    # ------------------------------------------------------------------
    # Other supported functions
    # ------------------------------------------------------------------
    def calculate_affinity_mat(self):
        Atotake = self.ligandindex.copy()
        Btotake = self.receptorindex.copy()
        allscores = self.scores.copy()
        for i in range(len(self.ligandindex)):
            if self.ligandindex[i] != self.receptorindex[i]:
                Atotake = np.append(Atotake, self.receptorindex[i])
                Btotake = np.append(Btotake, self.ligandindex[i])
                allscores = np.append(allscores, self.scores[i])
        A = self.TPM[Atotake, :]
        B = self.TPM[Btotake, :]
        affinitymat = (np.diag(allscores) @ A).T @ B
        return affinitymat

    def discretization(self, k):
        result = self.affinitymat.copy()
        n = self.affinitymat.shape[0]
        for i in range(n):
            row = self.affinitymat[i, :]
            I = np.argsort(row)[::-1]
            result[i, I[k:]] = 0
        result = (result + result.T) / 2.0
        return result

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------
    @staticmethod
    def identify_label(cells, labelpath):
        if os.path.isdir(labelpath):
            labelfile = os.path.join(labelpath, 'label.txt')
        elif os.path.isfile(labelpath):
            labelfile = labelpath
        else:
            raise FileNotFoundError('Cannot find label, check your labelpath')
        # Read as string to avoid numeric cell-name issues
        label_df = pd.read_csv(labelfile, sep='\t', header=None, names=['cell', 'label'], dtype=str)
        label_df['cell'] = label_df['cell'].astype(str).str.strip()
        label_df['label'] = label_df['label'].astype(str).str.strip()
        # Skip common header rows if present
        header_names = {'cell', 'cells', 'id', 'name', 'label', 'labels', 'cluster', 'type'}
        if label_df.iloc[0]['cell'].lower() in header_names or label_df.iloc[0]['label'].lower() in header_names:
            label_df = label_df.iloc[1:].reset_index(drop=True)
        cell_to_label = dict(zip(label_df['cell'], label_df['label']))
        slabels = [cell_to_label.get(c, 'unlabeled') for c in cells]
        standards = sorted(list(set(slabels)))
        ilabels = np.array([standards.index(l) + 1 for l in slabels])
        return ilabels, standards

    @staticmethod
    def getdistancemat(tsneresult):
        n = tsneresult.shape[0]
        sum_tsneresult = np.sum(tsneresult ** 2, axis=1, keepdims=True)
        distancemat = sum_tsneresult + sum_tsneresult.T - 2.0 * (tsneresult @ tsneresult.T)
        distancemat = np.sqrt(np.maximum(distancemat, 0))
        np.fill_diagonal(distancemat, np.inf)
        return distancemat

    @staticmethod
    def getplaindistance(tsneresult):
        n = tsneresult.shape[0]
        sum_tsneresult = np.sum(tsneresult ** 2, axis=1, keepdims=True)
        distancemat = sum_tsneresult + sum_tsneresult.T - 2.0 * (tsneresult @ tsneresult.T)
        distancemat = np.sqrt(np.maximum(distancemat, 0))
        plaindistance = []
        for i in range(n):
            for j in range(n):
                if j != i:
                    plaindistance.append([i + 1, j + 1, distancemat[i, j]])
        return np.array(plaindistance)
