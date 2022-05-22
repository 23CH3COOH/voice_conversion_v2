# -*- coding: utf-8 -*-
import os
import numpy as np
import pysptk
import pyworld
from scipy.io import wavfile
from align_mcep import align_mcep, align_mcep_dummy


def extract_f0_sp_ap(wav_path, FFT_SIZE):
    fs, data = wavfile.read(wav_path)
    data = data.astype(np.float)
    _f0, time = pyworld.dio(data, fs)
    f0 = pyworld.stonemask(data, _f0, time, fs)
    sp = pyworld.cheaptrick(data, f0, time, fs, fft_size=FFT_SIZE)
    ap = pyworld.d4c(data, _f0, time, fs, fft_size=FFT_SIZE)
    return f0, sp, ap

def resynthesis(wav_path_from, wav_path_to, out_wav_path, m, a, fs, FFT_SIZE):
    print('...Extracting by WORLD...')
    f0_from, sp_from, ap_from = extract_f0_sp_ap(wav_path_from, FFT_SIZE)
    f0_to, sp_to, ap_to = extract_f0_sp_ap(wav_path_to, FFT_SIZE)
    assert sp_to.shape == sp_from.shape
    assert ap_to.shape == ap_from.shape

    print('...Converting spectral envelope to mel cepstrum...')
    mcep_from = pysptk.sp2mc(sp_from, order=m, alpha=a)
    mcep_to = pysptk.sp2mc(sp_to, order=m, alpha=a)

    print('...Aligning mel cepstrum...')
    mcep_aligned_from, mcep_aligned_to = align_mcep_dummy(mcep_from, mcep_to)
    assert mcep_aligned_from.shape == mcep_aligned_to.shape

    print('...Converting mel cepstrum to spectral envelope...')
    sp_recalc_to = pysptk.mc2sp(mcep_aligned_to, alpha=a, fftlen=FFT_SIZE)
    assert sp_recalc_to.shape == sp_from.shape

    print('...Synthesizing...')
    synthesized = pyworld.synthesize(f0_from, sp_recalc_to, ap_from, fs)
    synthesized = synthesized.astype(np.int16)
    wavfile.write(out_wav_path, fs, synthesized)
    

if __name__ == '__main__':
    input_from = 'resynthesis_test_replacing_mcep/from/'
    input_to = 'resynthesis_test_replacing_mcep/to/'
    output = 'resynthesis_test_replacing_mcep/output/'
    m = 25
    a = 0.58
    fs = 48000
    FFT_SIZE = 1024

    for file_name in os.listdir(input_from):
        path_from = input_from + file_name
        path_to = input_to + file_name
        output_path = output + file_name
        if not os.path.exists(path_to):
            continue
        print('Start resynthesize: %s' % file_name)
        resynthesis(path_from, path_to, output_path, m, a, fs, FFT_SIZE)
