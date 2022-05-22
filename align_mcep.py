# -*- coding: utf-8 -*-
from scipy.spatial.distance import euclidean
from dtw import dtw


def align_mcep(mcep_1, mcep_2):
    dist, cost, dummy, path = dtw(mcep_1, mcep_2, dist=euclidean)
    aligned_mcep_1 = mcep_1[path[0]]
    aligned_mcep_2 = mcep_2[path[1]]
    return aligned_mcep_1, aligned_mcep_2
