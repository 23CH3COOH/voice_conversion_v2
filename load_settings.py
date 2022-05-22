# -*- coding: utf-8 -*-


def load_settings(settings_file_path='settings.txt'):
    f = open(settings_file_path)
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
