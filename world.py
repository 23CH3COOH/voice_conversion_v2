# -*- coding: utf-8 -*-
import numpy as np
import pyworld


def extract_sp(fs, signal, FFT_SIZE):
    signal = signal.astype(np.float)
    _f0, time = pyworld.dio(signal, fs)
    f0 = pyworld.stonemask(signal, _f0, time, fs)
    sp = pyworld.cheaptrick(signal, f0, time, fs, fft_size=FFT_SIZE)
    return sp

def extract_f0_sp_ap(fs, signal, FFT_SIZE):
    signal = signal.astype(np.float)
    _f0, time = pyworld.dio(signal, fs)
    f0 = pyworld.stonemask(signal, _f0, time, fs)
    sp = pyworld.cheaptrick(signal, f0, time, fs, fft_size=FFT_SIZE)
    ap = pyworld.d4c(signal, _f0, time, fs, fft_size=FFT_SIZE)
    return f0, sp, ap

def synthesize(f0, sp, ap, fs):
    synthesized = pyworld.synthesize(f0, sp, ap, fs)
    return synthesized.astype(np.int16)
