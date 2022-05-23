# -*- coding: utf-8 -*-
import numpy as np
import pysptk
from glob import glob
from scipy.io import wavfile
import world
from drawing_tools import draw_transition, draw_heatmap


f0_title_d = 'Fundamental frequency (%d frames)'
sp_title_dd = 'Spectral envelope (taken log10) (%d dimensions * %d frames)'
ap_title_dd = 'Aperiodicity (%d dimensions * %d frames)'
mc_title_dd = 'Mel cepstrum (%d dimensions * %d frames)'
wav_title_d = 'Wav signal (%d frames)'

'''
wav_pathから基本周波数とスペクトル包絡と非周期性指標を抽出し、音声を再合成する。
但し、スペクトル包絡は一旦メルケプストラムに変換し、再度スペクトル包絡に戻す。
'''
def resynthesis(wav_path, m, a, FFT_SIZE):
    print('...Extracting by WORLD...')
    fs, data = wavfile.read(wav_path)
    f0, sp, ap = world.extract_f0_sp_ap(fs, data, FFT_SIZE)

    np.savetxt(wav_path.replace('.wav', '_sp.txt'), sp, fmt ='%.6f')
    draw_transition(data,
                    wav_path.replace('.wav', '.png'),
                    wav_title_d % data.size)
    draw_transition(f0,
                    wav_path.replace('.wav', '_f0.png'),
                    f0_title_d % f0.size)
    draw_heatmap(np.log10(sp.T),
                 wav_path.replace('.wav', '_sp.png'),
                 sp_title_dd % (sp.shape[1], sp.shape[0]))
    draw_heatmap(ap.T,
                 wav_path.replace('.wav', '_ap.png'),
                 ap_title_dd % (ap.shape[1], ap.shape[0]))

    print('...Converting spectral envelope to mel cepstrum...')
    mcep = pysptk.sp2mc(sp, order=m, alpha=a)

    np.savetxt(wav_path.replace('.wav', '_mcep.txt'), mcep, fmt ='%.6f')
    draw_heatmap(mcep.T,
                 wav_path.replace('.wav', '_mcep.png'),
                 mc_title_dd % (mcep.shape[1], mcep.shape[0]))

    print('...Converting mel cepstrum to spectral envelope...')
    sp_r = pysptk.mc2sp(mcep, alpha=a, fftlen=FFT_SIZE)

    np.savetxt(wav_path.replace('.wav', '_sp_r.txt'), sp_r, fmt ='%.6f')
    draw_heatmap(np.log10(sp_r.T),
                 wav_path.replace('.wav', '_sp_r.png'),
                 sp_title_dd % (sp_r.shape[1], sp_r.shape[0]))

    print('...Synthesizing...')
    synthesized = world.synthesize(f0, sp_r, ap, fs)
    out_wav_path = wav_path.replace('.wav', '_resynthesized.wav')
    wavfile.write(out_wav_path, fs, synthesized)

    draw_transition(synthesized,
                    wav_path.replace('.wav', '_resynthesized.png'),
                    wav_title_d % synthesized.size)


if __name__ == '__main__':
    audio_folder = 'resynthesis_test/'
    m = 25
    a = 0.42
    FFT_SIZE = 1024
    for wav_path in glob(audio_folder + '*.wav'):
        if '_resynthesized.wav' in wav_path:
            continue
        print('Start resynthesize: %s' % wav_path)
        resynthesis(wav_path, m, a, FFT_SIZE)
