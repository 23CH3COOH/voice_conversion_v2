# -*- coding: utf-8 -*-
import os
import numpy as np
import pysptk
from scipy.io import wavfile
import world


'''
wav_path_fromの基本周波数と非周期性指標、wav_path_toのスペクトル包絡から
音声を再合成する。但し、wav_path_fromとwav_path_toで時間同期は取らない。
（時間同期を取るとフレーム数が変わってしまい、元のと揃えるのが難しいため。）
'''
def resynthesis(wav_path_from, wav_path_to, out_wav_path, m, a, FFT_SIZE):
    print('...Extracting by WORLD...')
    print(wav_path_from)
    fs, data = wavfile.read(wav_path_from)
    f0_from, sp_from, ap_from = world.extract_f0_sp_ap(fs, data, FFT_SIZE)
    fs, data = wavfile.read(wav_path_to)
    f0_to, sp_to, ap_to = world.extract_f0_sp_ap(fs, data, FFT_SIZE)
    assert sp_to.shape == sp_from.shape
    assert ap_to.shape == ap_from.shape

    print('...Converting spectral envelope to mel cepstrum...')
    mcep_to = pysptk.sp2mc(sp_to, order=m, alpha=a)

    print('...Converting mel cepstrum to spectral envelope...')
    sp_recalc_to = pysptk.mc2sp(mcep_to, alpha=a, fftlen=FFT_SIZE)
    assert sp_recalc_to.shape == sp_from.shape

    print('...Synthesizing...')
    synthesized = world.synthesize(f0_from, sp_recalc_to, ap_from, fs)
    wavfile.write(out_wav_path, fs, synthesized)


if __name__ == '__main__':
    input_from = 'resynthesis_test_replacing_mcep/from/'
    input_to = 'resynthesis_test_replacing_mcep/to/'
    output = 'resynthesis_test_replacing_mcep/output/'
    m = 25
    a = 0.58
    FFT_SIZE = 1024

    for file_name in os.listdir(input_from):
        if not '.wav' in file_name:
            continue
        path_to = input_to + file_name
        if not os.path.exists(path_to):
            continue
        path_from = input_from + file_name
        output_path = output + file_name
        print('Start resynthesize: %s' % file_name)
        resynthesis(path_from, path_to, output_path, m, a, FFT_SIZE)
