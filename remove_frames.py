# -*- coding: utf-8 -*-
import numpy as np
from sys import float_info


'''
メルケプストラムの全ての次元で同じ値が続くフレームを削除する
（但し、同じ値のフレームのうちの最後は削除されない）
'''
def remove_constant_frames(mcep_1, mcep_2):
    assert mcep_1.ndim == 2 and mcep_2.ndim == 2
    assert mcep_1.shape == mcep_2.shape
    assert np.all(np.isfinite(mcep_1)) and np.all(np.isfinite(mcep_2))

    diff_1 = np.sum(np.abs(np.diff(mcep_1, axis=0)), axis=1)
    remove_1 = np.where(diff_1 < float_info.min)[0]
    diff_2 = np.sum(np.abs(np.diff(mcep_2, axis=0)), axis=1)
    remove_2 = np.where(diff_2 < float_info.min)[0]

    remove_inds = np.sort(np.unique(np.concatenate([remove_1, remove_2])))
    mcep_1 = np.delete(mcep_1, remove_inds, axis=0)
    mcep_2 = np.delete(mcep_2, remove_inds, axis=0)
    return mcep_1, mcep_2, remove_inds

'''
メルケプストラムの第0次元の値がthreshold未満のフレームを削除する
'''
def remove_abnormally_low_frames(mcep_1, mcep_2, threshold):
    remove_1 = np.where(mcep_1[:, 0] < threshold)[0]
    remove_2 = np.where(mcep_2[:, 0] < threshold)[0]

    remove_inds = np.sort(np.unique(np.concatenate([remove_1, remove_2])))
    mcep_1 = np.delete(mcep_1, remove_inds, axis=0)
    mcep_2 = np.delete(mcep_2, remove_inds, axis=0)
    return mcep_1, mcep_2, remove_inds


if __name__ == '__main__':
    a = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                  [0.5, 1.5, 2.5, 3.5, 4.5],
                  [0.5, 1.5, 2.5, 3.5, 4.5],
                  [3.5, 5.5, 7.5, 8.5, 9.5]])
    b = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                  [0.0, 1.0, 2.0, 3.0, 4.0],
                  [2.5, 3.5, 4.5, 5.5, 6.5],
                  [-18, 5.5, 7.5, 8.5, 9.5]])
    a, b, remove_inds = remove_constant_frames(a, b)
    print(a, b, remove_inds)
    a, b, remove_inds = remove_abnormally_low_frames(a, b, -16)
    print(a, b, remove_inds)
