"""
Main running script for CSOmap.
Equivalent to MATLAB runme.m
Pure Python version - no .mat files.
"""
import os
import sys
import pickle
from preprocess import preprocess
from reconstruct_3d import reconstruct_3d
from analyst import Analyst
import matplotlib
matplotlib.use('Agg')


def runme(cancertype, condition='tobedetermined'):
    """
    Run all functions for a dataset.
    Data should be in "data/cancertype/" directory.
    Outputs saved in "output/cancertype/" directory.
    """
    datapath = os.path.join('data', cancertype)
    outputpath = os.path.join('output', cancertype)
    resultpath = os.path.join(outputpath, 'result')

    # Pre-process
    data_pkl = os.path.join(outputpath, 'data.pkl')
    if not os.path.exists(data_pkl):
        preprocess(datapath, outputpath)

    # Reconstruct 3D
    workspace_pkl = os.path.join(outputpath, 'workspace.pkl')
    if not os.path.exists(workspace_pkl):
        reconstruct_3d(outputpath, outputpath, 2, False, 50, condition)

    # Build analyst object
    a = Analyst(outputpath, datapath, resultpath, stat=True)

    # Save analyst as pickle
    analyst_pkl = os.path.join(outputpath, 'analyst.pkl')
    with open(analyst_pkl, 'wb') as f:
        pickle.dump(a, f)

    # Optionally clean up intermediate files (workspace.pkl is kept for further analysis)
    if os.path.exists(data_pkl):
        os.remove(data_pkl)
    # Keep workspace.pkl for in-silico experiments

    # Output basic results
    a.writeresult3d(f'{cancertype}_coordinate')
    a.writecounts(cancertype)
    a.writestatistics(f'{cancertype}_statistics')
    a.drawconclusion(0.05, f'{cancertype}_conclusion')
    a.savegif(f'{cancertype}_3dplot', f'{cancertype}_3dplot')
    a.mainLR(f'{cancertype}_mainLR')

    # Draw pictures (basic set)
    from draw_pictures import (
        draw_result3d_or_split_or_gif_with_gramm,
        draw_result3d_with_gramm,
        draw_sections_with_gramm,
        draw_bar_of_connection_number_with_gramm,
        draw_qvalue_with_gramm,
        draw_density_with_gramm,
        draw_result3d_with_section_with_gramm,
    )

    outputpath_result = os.path.join('output', cancertype, 'result')
    os.makedirs(outputpath_result, exist_ok=True)

    draw_result3d_or_split_or_gif_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_3d_global'), 'normal')
    draw_result3d_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_3d_views'), 0)
    draw_sections_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_sections_normal'), 'normal')
    draw_sections_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_sections_density'), 'density')
    draw_bar_of_connection_number_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_connection_number'), 0)
    draw_bar_of_connection_number_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_connection_number_normalized'), 1)
    draw_qvalue_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_qvalue'), a.standards,
                           150 / len(a.standards), 15 / (len(a.standards) ** 2), 15, 0.2, 0.2)
    draw_density_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_density'), 10, 2)
    draw_result3d_with_section_with_gramm(a, os.path.join(outputpath_result, f'{cancertype}_section_z=0'),
                                          [], [0, 90], [-float('inf'), float('inf')],
                                          [-float('inf'), float('inf')], [-5, 5], 'normal')
    print(f"CSOmap analysis for {cancertype} completed.")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ctype = sys.argv[1]
        cond = sys.argv[2] if len(sys.argv) > 2 else 'tobedetermined'
        runme(ctype, cond)
    else:
        print("Usage: python runme.py <cancertype> [condition]")
