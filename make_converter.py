# -*- coding: utf-8 -*-
import os
import pysptk
from align_mcep import align_mcep
from train_gmm import train_gmm
from world import extract_sp


wav_s = 'wav/train/%s/'
train_ss = 'train_result/%s_to_%s/'

settings_file = 'settings.txt'
wavf_s = '%s.wav'

class ConverterMaker:
    def __init__(self, conv_from, conv_to, output_visible_form=True):
        self.__from = conv_from
        self.__to = conv_to
        self.__output_visible_form = output_visible_form
        self.__train_files = list()
        self.__sp_from = list()
        self.__sp_to = list()
        self.__mcep_from = list()
        self.__mcep_to = list()
        self.__mcep_aligned_from = list()
        self.__mcep_aligned_to = list()

    def __load_settings(self):
        f = open(settings_file)
        rows = f.readlines()
        f.close()

        parsed = dict()
        for row in rows:
            if '#' in row or not '=' in row:
                continue
            splited = row.split('=')
            parsed[splited[0]] = splited[1].strip()

        self.__m = int(parsed['m'])
        self.__a = float(parsed['a'])
        self.__K = int(parsed['K'])
        self.__fft_size = int(parsed['fft_size'])

    def __search_common_wav_files(self):
        wav_files_from = os.listdir(wav_s % self.__from)
        wav_files_to = os.listdir(wav_s % self.__to)
        for file_name in wav_files_from:
            if '.wav' in file_name and file_name in wav_files_to:
                # 拡張子無しのファイル名リストの状態で保持しておく
                self.__train_files.append(file_name.replace('.wav', ''))

    def __extract_spectral_envelope(self):
        for file in self.__train_files:
            wave_path = wav_s % self.__from + wavf_s % file
            sp = extract_sp(wave_path, self.__fft_size)
            self.__sp_from.append(sp)
            wave_path = wav_s % self.__to + wavf_s % file
            sp = extract_sp(wave_path, self.__fft_size)
            self.__sp_to.append(sp)

    def __convert_mcep(self):
        assert len(self.__sp_from) == len(self.__sp_to)
        for sp in self.__sp_from:
            mcep = pysptk.sp2mc(sp, order=self.__m, alpha=self.__a)
            self.__mcep_from.append(mcep)
        for sp in self.__sp_to:
            mcep = pysptk.sp2mc(sp, order=self.__m, alpha=self.__a)
            self.__mcep_to.append(mcep)

    def __align_mcep(self):
        assert len(self.__mcep_from) == len(self.__mcep_to)
        for mcep_from, mcep_to in zip(self.__mcep_from, self.__mcep_to):
            mcep_aligned_from, mcep_aligned_to = align_mcep(mcep_from, mcep_to)
            self.__mcep_aligned_from.append(mcep_aligned_from)
            self.__mcep_aligned_to.append(mcep_aligned_to)
        
    def __train_gmm(self):
        assert len(self.__mcep_aligned_from) == len(self.__mcep_aligned_to)
        outdir = train_ss % (self.__from, self.__to)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        train_gmm(self.__mcep_aligned_from, self.__mcep_aligned_to, outdir,
                  self.__m, self.__K)

    def run(self):
        self.__load_settings()
        self.__search_common_wav_files()
        print('Extracting spectral envelope...')
        self.__extract_spectral_envelope()
        print('Conveting spectral envelope to mel cepstrum...')
        self.__convert_mcep()
        print('Aligning mel cepstrum...')
        self.__align_mcep()
        print('Training...')
        self.__train_gmm()


if __name__ == '__main__':
    converter_maker = ConverterMaker('clb', 'slt')
    converter_maker.run()