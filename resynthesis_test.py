# -*- coding: utf-8 -*-
import numpy as np
import pysptk
import pyworld
from glob import glob
from scipy.io import wavfile


def resynthesis(wav_path, m, a, FFT_SIZE):
    print('Extracting by WORLD...')
    fs, data = wavfile.read(wav_path)
    data = data.astype(np.float)
    _f0, time = pyworld.dio(data, fs)
    f0 = pyworld.stonemask(data, _f0, time, fs)
    sp = pyworld.cheaptrick(data, f0, time, fs, fft_size=FFT_SIZE)
    ap = pyworld.d4c(data, _f0, time, fs, fft_size=FFT_SIZE)
    print(f0.shape, sp.shape, ap.shape)

    print('Converting spectral envelope to mel cepstrum...')
    mcep_array = pysptk.sp2mc(sp, order=m, alpha=a)
    print(mcep_array.shape)

    print('Converting mel cepstrum to spectral envelope...')
    spectral_env_array = pysptk.mc2sp(mcep_array, alpha=a, fftlen=FFT_SIZE)
    print(spectral_env_array.shape)

    print('Synthesizing...')
    synthesized = pyworld.synthesize(f0, spectral_env_array, ap, fs)
    synthesized = synthesized.astype(np.int16)
    out_wav_path = wav_path.replace('.wav', '_resynthesized.wav')
    wavfile.write(out_wav_path, fs, synthesized)


if __name__ == '__main__':
    audio_folder = 'resynthesis_test/'
    m = 25
    a = 0.58
    FFT_SIZE = 1024
    for wav_path in glob(audio_folder + '*.wav'):
        if '_resynthesized.wav' in wav_path:
            continue
        print('Start resynthesize: %s' % wav_path)
        resynthesis(wav_path, m, a, FFT_SIZE)