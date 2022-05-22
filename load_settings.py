# -*- coding: utf-8 -*-
import os


def load_settings(conv_from, conv_to):
    m = 25
    a = 0.42
    K = 32
    fft_size = 1024
    
    path = 'settings/%s_to_%s/parameters.txt' % (conv_from, conv_to)
    if not os.path.exists(path):
        print('Not found settings file so apply default settings.')
        return m, a, K, fft_size

    f = open(path)
    rows = f.readlines()
    f.close()

    parsed = dict()
    for row in rows:
        if '#' in row or not '=' in row:
            continue
        splited = row.split('=')
        parsed[splited[0]] = splited[1].strip()

    m = int(parsed['m'])
    a = float(parsed['a'])
    K = int(parsed['K'])
    fft_size = int(parsed['fft_size'])
    return m, a, K, fft_size
