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

    #TR=======================================================================
    basename='/home/constantin/Desktop/Disertation/Codes/BCGN/BCGN-TR-29.06.2019-23:13:57'
    if len(sys.argv) > 2:
        basename2 = sys.argv[2]
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
    
    pn = measure.shape[0] #similar to size in matlab -problem
    sn = measure.shape[1] #solver
    rn = measure.shape[3] #runs
    Percentage_of_times_solved_TR=np.zeros([pn,sn,4])
    #solver order: 2D, N/2D, N-D, Adaptive
    for imetr, metr in enumerate(metrics):
        for p in range(pn):
            for s in range (sn):
                for r in range(rn):
                    if np.isnan(measure[p, s, imetr,r])==False:
                        Percentage_of_times_solved_TR[p,s,imetr]+=1
    Percentage_of_times_solved_TR[:,2,:]=Percentage_of_times_solved_TR[:,2,:]*rn#account for Gn being used only once
    #END TR -START REG========================================================
    Percentage_of_times_solved_TR=1/rn*Percentage_of_times_solved_TR
    #REG======================================================================
    basename='/home/constantin/Desktop/Disertation/Codes/BCGN/BCGN-REG-04.07.2019-13:12:45'
    if len(sys.argv) > 2:
        basename2 = sys.argv[2]
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
    
    pn = measure.shape[0] #similar to size in matlab -problem
    sn = measure.shape[1] #solver
    rn = measure.shape[3] #runs
    Percentage_of_times_solved_REG=np.zeros([pn,sn,4])
    #solver order: 2D, N/2D, N-D, Adaptive
    for imetr, metr in enumerate(metrics):
        for p in range(pn):
            for s in range (sn):
                for r in range(rn):
                    if np.isnan(measure[p, s, imetr,r])==False:
                        Percentage_of_times_solved_REG[p,s,imetr]+=1
    Percentage_of_times_solved_REG[:,2,:]=Percentage_of_times_solved_REG[:,2,:]*rn#account for Gn being used only once
    Percentage_of_times_solved_REG=1/rn*Percentage_of_times_solved_REG
    Diff= Percentage_of_times_solved_TR - Percentage_of_times_solved_REG

    filehandler=open('Jari_question.TR', 'wb')
    pickle.dump(Percentage_of_times_solved_TR, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open('Jari_question.REG', 'wb')
    pickle.dump(Percentage_of_times_solved_REG, filehandler, protocol=-1)
    filehandler.close()
    
    filehandler=open('Jari_question.DIFF', 'wb')
    pickle.dump(Diff, filehandler, protocol=-1)
    filehandler.close()
if __name__ == "__main__":
    main()