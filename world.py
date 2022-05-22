# -*- coding: utf-8 -*-
import numpy as np
import pyworld
from scipy.io import wavfile


def extract_sp(wav_path, FFT_SIZE):
    fs, data = wavfile.read(wav_path)
    data = data.astype(np.float)
    _f0, time = pyworld.dio(data, fs)
    f0 = pyworld.stonemask(data, _f0, time, fs)
    sp = pyworld.cheaptrick(data, f0, time, fs, fft_size=FFT_SIZE)
    return sp

def extract_f0_sp_ap(wav_path, FFT_SIZE):
    fs, data = wavfile.read(wav_path)
    data = data.astype(np.float)
    _f0, time = pyworld.dio(data, fs)
    f0 = pyworld.stonemask(data, _f0, time, fs)
    sp = pyworld.cheaptrick(data, f0, time, fs, fft_size=FFT_SIZE)
    ap = pyworld.d4c(data, _f0, time, fs, fft_size=FFT_SIZE)
    return f0, sp, ap
