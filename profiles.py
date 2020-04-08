""" Plot performance and budget profiles (treating each run as a 'separate function') """
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Dataset name and tolerance(s)
basename = 'BCGN-TR-NORMAL-HASHING'
tols = ['1e-01','1e-03','1e-05']

def main():

    for tol in tols:

        # Load data
        budget = pd.read_pickle(basename+'_'+tol+'.budget')
        print(budget)
        runtime = pd.read_pickle(basename+'_'+tol+'.runtime')
        print(runtime)

        #dimen = pd.read_pickle(basename+'.dimen')
        #print(dimen)
        #num_runs = budget.shape[0]/dimen.shape[1]
        #dimen = np.repeat(dimen.to_numpy(),num_runs)

        # Plot and save performance, budget and grad. eval. profiles
        fig_name = basename+'-'+tol
        performance_profile(budget,'Coordinate Evals Performance Profile',fig_name+'_coordevals','prof/')
        performance_profile(runtime,'Runtime Performance Profile',fig_name+'_runtime','prof/')
        #budget_profile(budget,np.array(dimen),fig_title,fig_name,'prof/')
        #grad_evals(budget,np.array(dimen),fig_title,fig_name,'evals/')

""" Calculate and Plot Performance Profile """
def performance_profile(measure, fig_title, fig_name, save_dir, tmax=50):
    """
    :param measure: prob x solver DataFrame,
     smallest values assumed to be the best
    """

    pn = measure.shape[0]
    sn = measure.shape[1]

    ratio = np.zeros((pn,sn))
    for p in range(pn):
        for s in range(sn):
            ratio[p,s] = measure.iloc[p,s] / np.min(measure.iloc[p,:])

    def profile(s,t):
        prob = 0
        for p in range(pn):
            if ratio[p,s] <= t:
                prob += 1
        return prob / pn

    t = np.linspace(1,tmax)
    prof = np.vectorize(profile)
    plt.figure(100)
    plt.clf()
    for s in range(sn):
        y = prof(s,t)
        plt.plot(t, y, '-', linewidth=2, clip_on=False)
    plt.grid()
    plt.xlabel('Performance Ratio')
    plt.ylabel('% Problems Solved')
    plt.legend(measure.columns, loc='lower right')
    if fig_title:
        plt.title(fig_title, fontsize=13)
    plt.tight_layout()

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name)


""" Calculate and Plot Budget Profile """
def budget_profile(measure, dimen, fig_title, fig_name, save_dir, bmax=50):
    """
    :param measure: prob x solver array,
     smallest values assumed to be the best
    """

    pn = measure.shape[0]
    sn = measure.shape[1]

    # scale budget by dimension
    ratio = np.zeros((pn,sn))
    for p in range(pn):
        for s in range(sn):
            ratio[p,s] = measure.iloc[p,s] / dimen[p]

    def profile(s,m):
        prob = 0
        for p in range(pn):
            if ratio[p,s] <= m:
                prob += 1
        return prob / pn

    m = np.linspace(0,bmax)
    prof = np.vectorize(profile)
    plt.figure(100)
    plt.clf()
    for s in range(sn):
        y = prof(s,m)
        plt.plot(m, y, '-', linewidth=2, clip_on=False)
    plt.grid()
    plt.xlabel('Budget')
    plt.ylabel('% Problems Solved')
    plt.legend(measure.columns, loc='lower right')
    if fig_title:
        plt.title(fig_title, fontsize=13)
    plt.tight_layout()

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name)

""" Plot gradient evaluations """
def grad_evals(measure, dimen, fig_title, fig_name, save_dir):
    """
    :param measure: prob x solver array,
     smallest values assumed to be the best
    """
    markers = ['v','D','s','o','^','*','x','+','d','.','3','>','<','1','2','4','8','|','_','h','p']

    pn = measure.shape[0]
    sn = measure.shape[1]

    # Scale by dimension to get gradient evals
    for p in range(pn):
        measure.iloc[p,:] /= dimen[p]

    plt.figure(100)
    plt.clf()
    for s in range(sn):
        plt.plot(np.arange(pn), measure.iloc[:,s], '.', marker=markers[s], markerSize=5, clip_on=False)

    plt.xticks(range(pn), measure.index, rotation=45, ha='right')
    plt.grid(alpha=0.5)
    plt.xlim([0,pn-1])
    plt.ylim([0,50])
    plt.ylabel('Grad. evals')
    plt.legend(measure.columns)
    if fig_title:
        plt.title(fig_title, fontsize=13)
    plt.tight_layout()

    if not os.path.exists(save_dir): os.makedirs(save_dir)
    plt.savefig(save_dir + '/' + fig_name)

main()
