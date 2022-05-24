# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt


def draw_transition(array, outpath, title, xlabel='frame', ylabel='', lw=0.1):
    fig = plt.figure(figsize=(7.2, 5.4))
    plt.plot(array, linewidth=lw)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(outpath)
    plt.clf()
    plt.close()

# 横長にできないが一旦諦める
def draw_heatmap(array, outpath, title, xlabel='frame', ylabel='dimension'):
    fig = plt.figure(figsize=(14.4, 10.8))
    ax = sns.heatmap(array, cmap='nipy_spectral')
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)  
    fig.savefig(outpath)
    plt.clf()
    plt.close()
