""" Plot performance and budget profiles """
from __future__ import division
from cycler import cycler
from palettable.colorbrewer import qualitative
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import os

# Timestamps of data
alg = 'tr'
timestamp = '03.08.2017-17:21:27'

def main():

    # Load data
    basename = 'BCGN-'+alg.upper()+'-'+timestamp
    print pickle.load(open(basename+'.funcs','rb'))
    dimen = pickle.load(open(basename+'.dimen','rb'))
    measure = pickle.load(open(basename+'.measure','rb'))
    labels = [r'$2$-BCGN',r'$\frac{n}{2}$-BCGN','GN',r'$2$-A-BCGN']
    metrics = ['budget: tau 1e-1','budget: tau 1e-3','budget: tau 1e-5','budget: tau 1e-7']

    # Plot performance profiles
    for imetr, metr in enumerate(metrics):
        fig_title = None
        save_dir = 'figures/'+alg.upper()+'/'+timestamp+'/'
        performance_profile(measure[:,:,imetr],labels,fig_title,metr,save_dir+'/perf/')
        budget_profile(measure[:,:,imetr],dimen,labels,fig_title,metr,save_dir+'/budget/')

""" Calculate and Plot Performance Profile """
def performance_profile(measure, solver_labels, fig_title, fig_name, save_dir):
    '''
    :param measure: prob x solver array,
     smallest values assumed to be the best
    '''

    # Set up colour brewer colours
    plt.rc('axes', prop_cycle=cycler('color', qualitative.Set1_9.mpl_colors))

    pn = measure.shape[0]
    sn = measure.shape[1]

    warnings.simplefilter("ignore", RuntimeWarning)
    ratio = np.zeros((pn, sn))
    for p in range(pn):
        for s in range(sn):
            ratio[p, s] = measure[p, s] / min(measure[p, :])
    warnings.resetwarnings()

    def profile(s, t):
        prob = 0
        for p in range(pn):
            if ratio[p, s] <= t:
                prob += 1
        return prob / pn

    t = np.linspace(1, 50)
    prof = np.vectorize(profile)
    plt.figure(100)
    plt.clf()
    for s in range(sn):
        y = prof(s, t)
        plt.plot(t, y, '-', linewidth=2, clip_on=False)
    plt.grid()
    plt.xlabel('Performance Ratio')
    plt.ylabel('% Problems Solved')
    plt.legend(solver_labels, loc='lower right')
    if fig_title:
        plt.title(fig_title, fontsize=13)

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name)


""" Calculate and Plot Budget Profile """
def budget_profile(measure, dimen, solver_labels, fig_title, fig_name, save_dir):
    '''
    :param measure: prob x solver array,
     smallest values assumed to be the best
    '''

    # Set up colour brewer colours
    plt.rc('axes', prop_cycle=cycler('color', qualitative.Set1_9.mpl_colors))

    pn = measure.shape[0]
    sn = measure.shape[1]

    # scale budget by dimension
    ratio = np.zeros((pn, sn))
    for p in range(pn):
        for s in range(sn):
            ratio[p, s] = measure[p, s] / dimen[p]

    def profile(s, m):
        prob = 0
        for p in range(pn):
            if ratio[p, s] <= m:
                prob += 1
        return prob / pn

    m = np.linspace(0, 50)
    prof = np.vectorize(profile)
    plt.figure(100)
    plt.clf()
    for s in range(sn):
        y = prof(s, m)
        plt.plot(m, y, '-', linewidth=2, clip_on=False)
    plt.grid()
    plt.xlabel('Budget')
    plt.ylabel('% Problems Solved')
    plt.legend(solver_labels, loc='lower right')
    if fig_title:
        plt.title(fig_title, fontsize=13)

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name)

main()
