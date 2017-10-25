""" Plot performance and budget profiles """
from __future__ import division
from cycler import cycler
from palettable.colorbrewer import qualitative
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import os

# Timestamp of data
alg = 'tr_approx'
timestamp = '04.10.2017-12:01:32'

def main():

    # Load data
    basename = 'BCGN-'+alg.upper()+'-'+timestamp
    funcs = pickle.load(open(basename+'.funcs','rb'))
    dimen = pickle.load(open(basename+'.dimen','rb'))
    measure = pickle.load(open(basename+'.measure','rb'))
    labels = pickle.load(open(basename + '.labels', 'rb'))
    metrics = ['budget: tau 1e-1','budget: tau 1e-3','budget: tau 1e-5','budget: tau 1e-7']

    # Plot and save performance, budget and grad. eval. profiles
    for imetr, metr in enumerate(metrics):
        fig_title = None
        performance_profile(measure[:,:,imetr],labels,fig_title,metr,'perf/')
        budget_profile(measure[:,:,imetr],np.array(dimen),labels,fig_title,metr,'budget/')
        grad_evals(measure[:,:,imetr],np.array(dimen),funcs,labels,fig_title,metr,'evals/')

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
    plt.tight_layout()

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
    plt.tight_layout()

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name)

""" Plot gradient evaluations """
def grad_evals(measure, dimen, prob_labels, solver_labels, fig_title, fig_name, save_dir):
    '''
    :param measure: prob x solver array,
     smallest values assumed to be the best
    '''

    # Set up colour brewer colours
    plt.rc('axes', prop_cycle=cycler('color', qualitative.Set1_9.mpl_colors))
    markers = ['v','D','s','o','^','*','x','+','d','.','3','>','<','1','2','4','8','|','_','h','p']

    # Scale by dimension to get gradient evals
    nfuncs = measure.shape[0]
    for f in range(nfuncs):
        measure[f,:] = measure[f,:]/dimen[f]

    plt.figure(100)
    plt.clf()
    for s in range(len(solver_labels)):
        plt.plot(np.arange(nfuncs), measure[:,s], '.', marker=markers[s], markerSize=5, clip_on=False)

    plt.xticks(range(nfuncs),prob_labels,rotation=45,ha='right')
    plt.grid(alpha=0.5)
    plt.xlim([0,nfuncs-1])
    plt.ylim([0,50])
    plt.ylabel('Grad. evals')
    plt.legend(solver_labels)
    if fig_title:
        plt.title(fig_title, fontsize=13)
    plt.tight_layout()

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name)

main()
