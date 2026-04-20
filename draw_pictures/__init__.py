"""
Drawing functions for CSOmap.
Equivalent to MATLAB draw_pictures/ functions, adapted to Python matplotlib.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.colors import ListedColormap
from PIL import Image


def draw_result3d_with_gramm(a, filename, split=False):
    """Draw 3D plot. If split, zoom-in to see each cluster."""
    if split:
        fig = plt.figure(figsize=(16, 8))
        nrows, ncols = 2, 3
    else:
        fig = plt.figure(figsize=(16, 6))
        nrows, ncols = 1, 3

    for i in range(2):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(a.standards)))
        for j in range(1, len(a.standards) + 1):
            toplot = a.result3d[a.labels == j, :]
            if len(toplot) == 0:
                continue
            ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                       c=[colors[j - 1, :3]], s=20, label=a.standards[j - 1])
        ax.set_title(f'angle: {-15 + 180 * i}')
        ax.view_init(elev=30, azim=-15 + 180 * i)
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_zticks([-40, -20, 0, 20, 40])

        if split:
            ax2 = fig.add_subplot(nrows, ncols, i + 4, projection='3d')
            splitresult3d = a.result3d.copy()
            center = splitresult3d.mean(axis=0)
            vectors = np.zeros((len(a.standards), 3))
            for j in range(1, len(a.standards) + 1):
                vectors[j - 1, :] = a.result3d[a.labels == j, :].mean(axis=0) - center
            vectors = vectors * 10
            for j in range(len(splitresult3d)):
                splitresult3d[j, :] += vectors[a.labels[j] - 1, :]
            for j in range(1, len(a.standards) + 1):
                toplot = splitresult3d[a.labels == j, :]
                if len(toplot) == 0:
                    continue
                ax2.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                            c=[colors[j - 1, :3]], s=20)
            ax2.set_title(f'split angle: {-15 + 180 * i}')
            ax2.view_init(elev=30, azim=-15 + 180 * i)

    ax = fig.add_subplot(nrows, ncols, ncols if not split else 3, projection='3d')
    for j in range(1, len(a.standards) + 1):
        toplot = a.result3d[a.labels == j, :]
        if len(toplot) == 0:
            continue
        marker = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][j % 10]
        ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                   c=[colors[j - 1, :3]], marker=marker, s=20, label=a.standards[j - 1])
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_result3d_or_split_or_gif_with_gramm(a, filename, option='normal'):
    if option == 'normal':
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(a.standards)))
        for j in range(1, len(a.standards) + 1):
            toplot = a.result3d[a.labels == j, :]
            if len(toplot) == 0:
                continue
            marker = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][j % 10]
            ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                       c=[colors[j - 1, :3]], marker=marker, s=30, label=a.standards[j - 1])
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_zticks([-40, -20, 0, 20, 40])
        ax.view_init(elev=30, azim=20)
        plt.tight_layout()
        fig.savefig(f'{filename}_origin.pdf', dpi=150)
        plt.close(fig)
    elif option == 'split':
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        splitresult3d = a.result3d.copy()
        center = splitresult3d.mean(axis=0)
        vectors = np.zeros((len(a.standards), 3))
        for j in range(1, len(a.standards) + 1):
            vectors[j - 1, :] = a.result3d[a.labels == j, :].mean(axis=0) - center
        vectors = vectors * 2
        for j in range(len(splitresult3d)):
            splitresult3d[j, :] += vectors[a.labels[j] - 1, :]
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(a.standards)))
        for j in range(1, len(a.standards) + 1):
            toplot = splitresult3d[a.labels == j, :]
            if len(toplot) == 0:
                continue
            ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                       c=[colors[j - 1, :3]], s=20)
        ax.view_init(elev=60, azim=120)
        plt.tight_layout()
        fig.savefig(f'{filename}_split.pdf', dpi=150)
        plt.close(fig)
    elif option == 'gif':
        images = []
        for i in range(36):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            cmap = plt.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, len(a.standards)))
            for j in range(1, len(a.standards) + 1):
                toplot = a.result3d[a.labels == j, :]
                if len(toplot) == 0:
                    continue
                ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                           c=[colors[j - 1, :3]], s=20)
            ax.view_init(elev=30, azim=10 * i + 5)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())
            img = img[:, :, :3]
            images.append(img)
            plt.close(fig)
        if len(images) > 0:
            gif_path = f'{filename}.gif'
            images_pil = [Image.fromarray(im) for im in images]
            images_pil[0].save(gif_path, save_all=True, append_images=images_pil[1:], duration=250, loop=0)


def draw_density_with_gramm(a, filename, width1=3, width2=0.5):
    density = np.zeros(len(a.cells), dtype=int)
    for i in range(len(a.cells)):
        for j in range(len(a.standards)):
            density[i] += a.neighbor[i][j].shape[0]
    labels = [a.standards[l - 1] for l in a.labels]
    df = {'label': labels, 'density': np.log(density + 1)}
    import pandas as pd
    df = pd.DataFrame(df)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='label', y='density', data=df, ax=ax, inner=None, cut=0, scale='width', width=width1)
    sns.boxplot(x='label', y='density', data=df, ax=ax, width=width2, showcaps=False,
                boxprops={'facecolor': 'None'}, showfliers=False, whiskerprops={'linewidth': 0})
    ax.set_title('Density')
    ax.set_ylabel('log(density+1)')
    ax.set_xlabel('')
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_bar_of_connection_number_with_gramm(a, filename, normalize=False):
    connectionnumber = np.zeros(len(a.standards))
    for i in range(len(a.counts)):
        for j in range(len(a.counts[i])):
            connectionnumber[i] += a.counts[i][j].shape[0]
    if normalize:
        y = connectionnumber / (a.clustercounts + 1e-10)
        ylabel = 'number (normalized)'
    else:
        y = connectionnumber
        ylabel = 'number'
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(a.standards)), y, color=sns.color_palette('tab20', len(a.standards)))
    ax.set_xticks(range(len(a.standards)))
    ax.set_xticklabels(a.standards, rotation=30, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title('Connection number')
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_qvalue_with_gramm(a, filename, legends=None, pointsize=10, step=1,
                           textsize=12, leftedge=0.1, bottomedge=0.1,
                           mainposition=None):
    if legends is None:
        legends = a.standards
    if mainposition is None:
        mainposition = [0, 0, 0.85, 0.85]
    order = [a.standards.index(l) for l in legends if l in a.standards]
    connection = a.connection[np.ix_(order, order)]
    reverseconnection = a.reverseconnection[np.ix_(order, order)]
    counts = [[a.counts[o1][o2] for o2 in order] for o1 in order]

    positionX, positionY, S, C = [], [], [], []
    for i in range(len(order)):
        for j in range(len(order)):
            positionX.append(j + 1)
            positionY.append(i + 1)
            if connection[i, j] <= 0.05:
                C.append('enriched')
            elif reverseconnection[i, j] <= 0.05:
                C.append('depleted')
            else:
                C.append('other')
            S.append(counts[i][j].shape[0])

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 6, 1], height_ratios=[1, 6, 1],
                          left=leftedge, bottom=bottomedge,
                          wspace=0.05, hspace=0.05)

    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.bar(range(1, len(order) + 1), a.clustercounts[order], color='steelblue', edgecolor='white')
    ax_top.set_xticks([])
    ax_top.set_xlim(0.5, len(order) + 0.5)
    ax_top.set_ylabel('cell counts')
    ax_top.spines['bottom'].set_visible(False)

    ax_right = fig.add_subplot(gs[1, 2])
    ax_right.barh(range(1, len(order) + 1), a.clustercounts[order], color='steelblue', edgecolor='white')
    ax_right.set_yticks([])
    ax_right.set_ylim(0.5, len(order) + 0.5)
    ax_right.invert_yaxis()
    ax_right.set_ylabel('cell counts')
    ax_right.spines['left'].set_visible(False)

    ax_main = fig.add_subplot(gs[1, 1])
    color_map = {'enriched': '#E74C3C', 'other': '#95A5A6', 'depleted': '#3498DB'}
    for ctype in ['enriched', 'other', 'depleted']:
        mask = [c == ctype for c in C]
        ax_main.scatter(np.array(positionX)[mask], np.array(positionY)[mask],
                        s=np.array(S)[mask] * step + pointsize,
                        c=color_map[ctype], label=ctype, alpha=0.8)
    ax_main.set_xticks(range(1, len(order) + 1))
    ax_main.set_xticklabels(legends, rotation=30, ha='right')
    ax_main.set_yticks(range(1, len(order) + 1))
    ax_main.set_yticklabels(legends)
    ax_main.set_xlim(0.5, len(order) + 0.5)
    ax_main.set_ylim(0.5, len(order) + 0.5)
    ax_main.invert_yaxis()
    ax_main.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_sections_with_gramm(a, filename, mode='normal'):
    for i in range(-50, 31, 20):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(a.standards)))
        for j in range(1, len(a.standards) + 1):
            toplot = a.result3d[a.labels == j, :]
            if len(toplot) == 0:
                continue
            ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                       c=[colors[j - 1, :3]], s=20, label=a.standards[j - 1])
        ax.set_zlim(i, i + 20)
        ax.view_init(elev=0, azim=0)
        ax.set_title(f"z range: [{i}, {i + 20}]")
        plt.tight_layout()
        fig.savefig(f"{filename}_z=[{i},{i+20}].pdf", dpi=150)
        plt.close(fig)


def draw_result3d_with_section_with_gramm(a, filename, legendorder=None, view_angles=None, xlim=None, ylim=None, zlim=None, mode='normal'):
    if view_angles is None or len(view_angles) == 0:
        view_angles = [0, 90]
    if xlim is None:
        xlim = [-float('inf'), float('inf')]
    if ylim is None:
        ylim = [-float('inf'), float('inf')]
    if zlim is None:
        zlim = [-float('inf'), float('inf')]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if mode == 'normal':
        cmap = plt.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, len(a.standards)))
        order = list(range(1, len(a.standards) + 1))
        if legendorder is not None and len(legendorder) > 0:
            order = [a.standards.index(l) + 1 for l in legendorder if l in a.standards]
        for j in order:
            toplot = a.result3d[a.labels == j, :]
            if len(toplot) == 0:
                continue
            ax.scatter(toplot[:, 0], toplot[:, 1], toplot[:, 2],
                       c=[colors[j - 1, :3]], s=3, label=a.standards[j - 1])
        ax.set_title('3d global' if zlim[0] == -float('inf') else f"z = {(zlim[0]+zlim[1])/2:.1f}")
    elif mode == 'density':
        density = np.zeros(len(a.cells), dtype=int)
        for i in range(len(a.cells)):
            for j in range(len(a.standards)):
                density[i] += a.neighbor[i][j].shape[0]
        sc = ax.scatter(a.result3d[:, 0], a.result3d[:, 1], a.result3d[:, 2],
                        c=np.log2(density + 1), cmap='autumn', s=3)
        plt.colorbar(sc, ax=ax, shrink=0.5, label='density')
        ax.set_title('3d global' if zlim[0] == -float('inf') else f"z = {(zlim[0]+zlim[1])/2:.1f}")
    # Replace inf limits with actual data bounds (matplotlib 3D does not support inf)
    if any(np.isinf(xlim)):
        xlim = [a.result3d[:, 0].min(), a.result3d[:, 0].max()]
    if any(np.isinf(ylim)):
        ylim = [a.result3d[:, 1].min(), a.result3d[:, 1].max()]
    if any(np.isinf(zlim)):
        zlim = [a.result3d[:, 2].min(), a.result3d[:, 2].max()]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=view_angles[1], azim=view_angles[0])
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_result3d_with_genes_with_gramm(a, filename, view_angles, xlim, ylim, zlim, genes):
    if isinstance(genes, str):
        genes = [genes]
    if view_angles is None or len(view_angles) == 0:
        view_angles = [30, 30]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Plot background cells in gray
    ax.scatter(a.result3d[:, 0], a.result3d[:, 1], a.result3d[:, 2],
               c='lightgray', s=10, alpha=0.3)
    # Highlight gene-expressing cells
    for gene in genes:
        if gene not in a.genes:
            continue
        gene_idx = a.genes.index(gene)
        expr = a.TPM[gene_idx, :]
        mask = expr > 0
        sc = ax.scatter(a.result3d[mask, 0], a.result3d[mask, 1], a.result3d[mask, 2],
                        c=expr[mask], cmap='Reds', s=30)
        plt.colorbar(sc, ax=ax, shrink=0.5, label=gene)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=view_angles[1], azim=view_angles[0])
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_one_gene_with_gramm(a, gene, filename, pointsize=10, threshold=0.5):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    gene_idx = a.genes.index(gene)
    expr = np.log2(a.TPM[gene_idx, :] + 1)
    sc = ax.scatter(a.result3d[:, 0], a.result3d[:, 1], a.result3d[:, 2],
                    c=expr, cmap='viridis', s=pointsize)
    plt.colorbar(sc, ax=ax, shrink=0.5, label=gene)
    ax.set_title(f'Expression of {gene}')
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_compare_one_gene_with_gramm(a1, a2, name1, cluster1, name2, cluster2,
                                     gene, filename, pointsize=10, threshold=0.2):
    fig = plt.figure(figsize=(16, 6))
    for idx, (a, name, cluster) in enumerate([(a1, name1, cluster1), (a2, name2, cluster2)]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        gene_idx = a.genes.index(gene)
        expr = np.log2(a.TPM[gene_idx, :] + 1)
        cidx = a.standards.index(cluster) + 1
        mask = a.labels == cidx
        ax.scatter(a.result3d[:, 0], a.result3d[:, 1], a.result3d[:, 2],
                   c='lightgray', s=pointsize, alpha=0.3)
        sc = ax.scatter(a.result3d[mask, 0], a.result3d[mask, 1], a.result3d[mask, 2],
                        c=expr[mask], cmap='Reds', s=pointsize * 2)
        plt.colorbar(sc, ax=ax, shrink=0.5)
        ax.set_title(f'{name} {cluster} {gene}')
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_compare_density_with_gramm(a1, a2, clusters, label1, label2, filename,
                                    width1=1, width2=0.1, pair=None):
    densities = []
    labels_list = []
    groups = []
    for a, group_label in [(a1, label1), (a2, label2)]:
        for i in range(len(a.cells)):
            d = 0
            for j in range(len(a.standards)):
                d += a.neighbor[i][j].shape[0]
            densities.append(d)
            labels_list.append(a.standards[a.labels[i] - 1])
            groups.append(group_label)
    import pandas as pd
    df = pd.DataFrame({'density': np.log(np.array(densities) + 1),
                       'label': labels_list, 'group': groups})
    if clusters and len(clusters) > 0:
        df = df[df['label'].isin(clusters)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='label', y='density', hue='group', data=df, ax=ax,
                   split=True if pair else False, inner='quart', scale='width')
    ax.set_ylabel('log(density+1)')
    ax.set_xlabel('')
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_compare_of_connection_number_with_gramm(a1, a2, clusters1, clusters2,
                                                 label1, label2, filename, mode='density'):
    cidx1_a1 = a1.standards.index(clusters1[0])
    cidx2_a1 = a1.standards.index(clusters1[1])
    cidx1_a2 = a2.standards.index(clusters2[0])
    cidx2_a2 = a2.standards.index(clusters2[1])

    count1 = a1.counts[cidx1_a1][cidx2_a1].shape[0]
    count2 = a2.counts[cidx1_a2][cidx2_a2].shape[0]

    if mode == 'normalized_number':
        y1 = count1 / (a1.clustercounts[cidx1_a1] + 1e-10)
        y2 = count2 / (a2.clustercounts[cidx1_a2] + 1e-10)
        ylabel = 'normalized number'
    else:
        y1 = count1
        y2 = count2
        ylabel = 'number'

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar([label1, label2], [y1, y2], color=['steelblue', 'coral'])
    ax.set_ylabel(ylabel)
    ax.set_title(f'{clusters1[0]}---{clusters1[1]} connection')
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def draw_pseudo_with_gramm(data1, data2, label1, label2, filename, scale='linear'):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(data1.shape[0])
    if scale == 'log':
        ax.semilogy(x, data1, label=label1, marker='o')
        ax.semilogy(x, data2, label=label2, marker='s')
    else:
        ax.plot(x, data1, label=label1, marker='o')
        ax.plot(x, data2, label=label2, marker='s')
    ax.legend()
    ax.set_xlabel('Pseudo step')
    ax.set_ylabel('Value')
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)


def correlation_two_genes(a, clusterA, clusterB, genes, limits, title, filename):
    from scipy import stats
    cA = a.standards.index(clusterA) + 1
    cB = a.standards.index(clusterB) + 1
    cellsA = np.where(a.labels == cA)[0]
    cellsB = np.where(a.labels == cB)[0]

    g1_idx = a.genes.index(genes[0])
    g2_idx = a.genes.index(genes[1])
    expr1 = a.TPM[g1_idx, cellsA]
    expr2 = a.TPM[g2_idx, cellsB]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(expr1, expr2, alpha=0.5)
    ax.set_xlabel(genes[0])
    ax.set_ylabel(genes[1])
    ax.set_title(title)
    if limits:
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    plt.tight_layout()
    fig.savefig(f'{filename}.pdf', dpi=150)
    plt.close(fig)

    # Simple correlation using all cells (approximate)
    common = min(len(expr1), len(expr2))
    r, p = stats.pearsonr(expr1[:common], expr2[:common])
    return r, p


def draw_for_one_dataset(a, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    draw_result3d_or_split_or_gif_with_gramm(a, os.path.join(outdir, f'{name}_3d_global'), 'normal')
    draw_result3d_with_gramm(a, os.path.join(outdir, f'{name}_3d_views'), 0)
    draw_sections_with_gramm(a, os.path.join(outdir, f'{name}_sections_normal'), 'normal')
    draw_sections_with_gramm(a, os.path.join(outdir, f'{name}_sections_density'), 'density')
    draw_bar_of_connection_number_with_gramm(a, os.path.join(outdir, f'{name}_connection_number'), 0)
    draw_bar_of_connection_number_with_gramm(a, os.path.join(outdir, f'{name}_connection_number_normalized'), 1)
    draw_density_with_gramm(a, os.path.join(outdir, f'{name}_density'), 3, 0.5)
    draw_qvalue_with_gramm(a, os.path.join(outdir, f'{name}_qvalue'), a.standards,
                           150 / len(a.standards), 15 / (len(a.standards) ** 2), 15, 0.2, 0.2)


def draw_all_pictures():
    """
    Equivalent to MATLAB draw_all_pictures.m
    Loads pre-computed analyst objects and reproduces paper figures.
    """
    print("draw_all_pictures: This is a template. Please load your analyst objects and call drawing functions accordingly.")
