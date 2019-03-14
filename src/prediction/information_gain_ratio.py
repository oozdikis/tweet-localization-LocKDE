'''
@author Ozer Ozdikis
@license:  See 'LICENSE.md' as part of this package. 
@precondition:
@summary: This file includes the functions to calculate information gain ratio for tokens. 
'''

import numpy as np
from math import log

def get_entrophy(grid_assignments):
    gc_counts = np.bincount(grid_assignments)
    denominator = float(len(grid_assignments))
    p = gc_counts[np.nonzero(gc_counts)] / denominator 
    return - np.sum(p * np.log(p))

def find_inf_gain_ratio(tgms_training, grid_entropy, token):
    document_count = len(tgms_training)
    token_exists_gridcells = []
    token_not_exists_gridcells = []
    for tgm in tgms_training:
        if token in tgm.tokens:
            token_exists_gridcells.append(tgm.gcid)
        else:
            token_not_exists_gridcells.append(tgm.gcid)
    token_exists_tgm_count = len(token_exists_gridcells)
    token_not_exists_tgm_count = len(token_not_exists_gridcells)
    entropy_token_exists = get_entrophy(token_exists_gridcells)
    entropy_token_not_exists = get_entrophy(token_not_exists_gridcells)
    inf_gain = grid_entropy - (((1.0 * token_exists_tgm_count / document_count) * entropy_token_exists)
                             + ((1.0 * token_not_exists_tgm_count / document_count) * entropy_token_not_exists))
    pw = 1.0 * token_exists_tgm_count / document_count
    pwx = 1.0 * token_not_exists_tgm_count / document_count
    intrinsic_entrophy = - pw * log(pw) - pwx * log(pwx)
    igr = inf_gain / intrinsic_entrophy
    return igr

def find_inf_gain_ratios(tgms_training, tokens_list):
    inf_gain_ratios = {}
    grid_assignments = [tgm.gcid for tgm in tgms_training]
    grid_entropy = get_entrophy(grid_assignments)
    for token in tokens_list:
        igr = find_inf_gain_ratio(tgms_training, grid_entropy, token)
        inf_gain_ratios[token] = igr
    return inf_gain_ratios



