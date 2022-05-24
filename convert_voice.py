# -*- coding: utf-8 -*-
import os
import numpy as np
import joblib
import pysptk
from scipy.io import wavfile
from scipy.stats import multivariate_normal
import world
from load_settings import load_settings


train_ss = 'train_result/%s_to_%s/'
wav_s = 'wav/production/%s/'
gmm_file = 'GMM.gmm'

# 式(9)の計算
def P(k, x, gmm, gauss, denom):
    return denom[k] / np.sum(denom)

# 式(11)の計算
def E(k, x, gmm, ss, m):
    d = m + 1
    return gmm.means_[k, d:] + np.dot(ss[k], x - gmm.means_[k, 0:d])

# 式(13)の計算
def convert_frame(x, gmm, gauss, ss, m, K):
    # 式(9)の分母だけ先に計算
    denom = np.zeros(K)
    for n in range(K):
        denom[n] = gmm.weights_[n] * gauss[n].pdf(x)

    y = np.zeros_like(x)
    for k in range(K):
        y += P(k, x, gmm, gauss, denom) * E(k, x, gmm, ss, m)
    return y

def convert_mcep(mcep, gmm, m, K):
    d = m + 1
    res = np.full(mcep.shape, np.nan)

    # 式(9)の多次元正規分布のオブジェクトを作成しておく
    gauss = []
    for k in range(K):
        gauss.append(multivariate_normal(gmm.means_[k, 0:d],
                                         gmm.covariances_[k, 0:d, 0:d]))

    # 式(11)のフレームtに依存しない項を計算しておく
    ss = []
    for k in range(K):
        ss.append(np.dot(gmm.covariances_[k, d:, 0:d],
                         np.linalg.inv(gmm.covariances_[k, 0:d, 0:d])))

    # 各フレームをGMMで変形する
    for t in range(len(mcep)):
        res[t] = convert_frame(mcep[t], gmm, gauss, ss, m, K)

    assert np.all(np.isfinite(res))
    return res

def convert_voice(wav_path_from, wav_path_to, gmm_path, m, a, K, FFT_SIZE):
    fs, data = wavfile.read(wav_path_from)
    f0, sp, ap = world.extract_f0_sp_ap(fs, data, FFT_SIZE)
    mcep = pysptk.sp2mc(sp, order=m, alpha=a)
    mcep_to = convert_mcep(mcep, joblib.load(gmm_path), m, K)
    sp_to = pysptk.mc2sp(mcep_to, alpha=a, fftlen=FFT_SIZE)
    synthesized = world.synthesize(f0, sp_to, ap, fs)
    wavfile.write(wav_path_to, fs, synthesized)

def main(conv_from, conv_to):
    gmm_path = train_ss % (conv_from, conv_to) + gmm_file
    if not os.path.exists(gmm_path):
        print('Not found train result.')
        return

    m, a, K, FFT_SIZE = load_settings(conv_from, conv_to)
    if not os.path.exists(wav_s % conv_to):
        os.makedirs(wav_s % conv_to)

    for file in os.listdir(wav_s % conv_from):
        wav_path_from = wav_s % conv_from + file
        wav_path_to = wav_s % conv_to + file
        convert_voice(wav_path_from, wav_path_to, gmm_path, m, a, K, FFT_SIZE)


if __name__ == '__main__':
    main('hatsune', 'riko')
