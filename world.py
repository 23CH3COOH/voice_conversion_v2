# -*- coding: utf-8 -*-
import numpy as np
import pyworld


'''
【引数】
 - fs: サンプリング周波数[Hz]
 - signal: Wavファイルに記録されているデータ（scipy.io.wavfile.readの戻り値）
 - FFT_SIZE: スペクトル包絡計算時に何個でDFTをとるか
 - ratio: signalを一律何倍してから変換するか
【戻り値】
 - f0: 基本周波数（1次元ndarray）
 - sp: スペクトル包絡（shapeが(n, 1 + FFT_SIZE // 2)の2次元ndarray）
 - ap: 非周期性指標（shapeが(n, 1 + FFT_SIZE // 2)の2次元ndarray）
'''
def extract_sp(fs, signal, FFT_SIZE, ratio=1.0):
    signal = ratio * signal.astype(np.float)
    _f0, time = pyworld.dio(signal, fs)
    f0 = pyworld.stonemask(signal, _f0, time, fs)
    sp = pyworld.cheaptrick(signal, f0, time, fs, fft_size=FFT_SIZE)
    return sp

def extract_f0_sp_ap(fs, signal, FFT_SIZE, ratio=1.0):
    signal = ratio * signal.astype(np.float)
    _f0, time = pyworld.dio(signal, fs)
    f0 = pyworld.stonemask(signal, _f0, time, fs)
    sp = pyworld.cheaptrick(signal, f0, time, fs, fft_size=FFT_SIZE)
    ap = pyworld.d4c(signal, _f0, time, fs, fft_size=FFT_SIZE)
    return f0, sp, ap

'''
音声を合成する（scipy.io.wavfile.writeに与えられる配列を返す）
'''
def synthesize(f0, sp, ap, fs):
    synthesized = pyworld.synthesize(f0, sp, ap, fs)
    return synthesized.astype(np.int16)
