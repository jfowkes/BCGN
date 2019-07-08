""" Plot performance and budget profiles """
from __future__ import division
from cycler import cycler
from palettable.colorbrewer import qualitative
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
import sys
import os
import math

# Dataset name(s)
# basename = sys.argv[0] #use this to sort of get path then cut and paste the begning of the name
#and then write it in basename fct
basename='/home/constantin/Desktop/Disertation/Codes/BCGN/BCGN-TR-06.07.2019-17:50:27'
if len(sys.argv) > 2:
    basename2 = sys.argv[2]

def main():

    # Load data(s)
    funcs = pickle.load(open(basename+'.funcs','rb'))
    dimen = pickle.load(open(basename+'.dimen','rb'))
    measure1 = pickle.load(open(basename+'.measure','rb')) #3 dim - runs coordinate is missing apparently
    if len(sys.argv) > 2:
        measure2 = pickle.load(open(basename2+'.measure','rb')) 
        measure = np.hstack((measure1,measure2))
    else:
        measure = measure1
    labels1 = pickle.load(open(basename+'.labels','rb'))
    if len(sys.argv) > 2:
        labels2 = pickle.load(open(basename2+'.labels','rb'))
        #labels2 = [l+' beta' for l in labels2]
        labels = np.hstack((labels1,labels2))
    else:
        labels = labels1
    metrics = ['budget: tau 1e-1','budget: tau 1e-3','budget: tau 1e-5','budget: tau 1e-7']

    # Plot and save performance, budget and grad. eval. profiles
    for imetr, metr in enumerate(metrics):
        fig_title = None
        budget_profile(measure[:,:,imetr,:],np.array(dimen),labels,fig_title,metr,'budget/')
        performance_profile(measure[:,:,imetr,:],labels,fig_title,metr,'perf/')
 #       grad_evals(measure[:,:,imetr,:],np.array(dimen),funcs,labels,fig_title,metr,'evals/')

""" Calculate and Plot Performance Profile """
def performance_profile(measure, solver_labels, fig_title, fig_name, save_dir):
    '''
    :param measure: prob x solver array,
     smallest values assumed to be the best
    '''
    # Set up colour brewer colours
    plt.rc('axes', prop_cycle=cycler('color', qualitative.Set1_9.mpl_colors))

    pn = measure.shape[0] #similar to size in matlab -problem
    sn = measure.shape[1] #solver
    rn = measure.shape[2]#runs
     #NAN IS TREATED AS ZERO WHEN MEAN IS APPLIED!!!
    #THIS CAUSES ALL min(P,:)=nan WHENEVER THERE EXISTS A SOLVER S THAT DOES NOT SOLVE
    #PROBLEM P
    #THAT IS WHY ALL SOLVERS BASICALLY SOLVE the SAME NUMBER OF PROBLEMS!!!
    #NEED TO TURN IT TO 10^16 !!!!!!!
   # for pii in range(pn):
    #    for sii in range(sn):
     #       if (math.isnan(measure[pii,sii])):
      #          measure[pii,sii]=5*pow(10,15)

    warnings.simplefilter("ignore", RuntimeWarning) #pn is the problem index, 
    #sn is the solver (no of coordinate/ kappa value -that's why they are put together)
    ratio = np.zeros((pn*rn, sn))
    for p in range(pn):
        for r in range (rn):
            for s in range(sn):
                ratio[p*rn+r, s] = measure[p, s, r] / np.nanmin(measure[p, :, :])
    warnings.resetwarnings()

    def profile(s, t): # function which computes percentage of problems solved
        prob = 0
        for pr in range(pn*rn): 
            if ratio[pr, s] <= t:
                prob += 1
        return prob / pn/rn

    t = np.linspace(1, 50)
    prof = np.vectorize(profile) #vectorize the function above to make it work like
    #Matlab maths rather than concatnation
    plt.figure(figsize=(10.0, 6))
    plt.clf()
    for s in range(sn):
        y = prof(s, t)
        if s==2: #if we do the GN
            y=y*rn #multiply by the number of runs to account for the fact that we only solve once
        plt.plot(t, y, '-', linewidth=2, clip_on=False)
    plt.grid()
    plt.ylim([0,1])
    plt.xlabel('Performance Ratio')
    plt.ylabel('% of Problems Solved')#Number of Problems Solved: Total 32x40=1280
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
    rn = measure.shape[2]#runs
    # scale budget by dimension
    ratio = np.zeros((pn*rn, sn))
    for p in range(pn):
        for r in range (rn):
            for s in range(sn):
                ratio[p*rn+r, s] = measure[p, s, r] / dimen[p]#/dimen[p]

    def profile(s, m):
        prob = 0
        for pr in range(pn*rn):
            if ratio[pr, s] <= m:
                prob += 1
        return prob / pn/rn

    m = np.linspace(0, 50)#**2
    prof = np.vectorize(profile)
    plt.figure(figsize=(10,6))
    plt.clf()
    for s in range(sn):
        y = prof(s, m) #!!!!!!!!!!!!!!
        if s==2:
            y=y*rn #if we deal with full GN, multiply by rn to account fo the fact
            #that we only solve once
        plt.plot(m, y, '-', linewidth=2, clip_on=False)
    plt.grid()
    plt.ylim([0,1])
    plt.xlabel('Budget')
    plt.ylabel('% of Problems Solved')#Number of Problems Solved: Total 32x40=1280
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
