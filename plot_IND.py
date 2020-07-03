""" Individual function plots for paper """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Test functions
#funcs = ['chemotherapy','gisette']
#dimen = [61359,5000]
funcs = ['OSCIGRNE','ARTIF','BRATU2D']
dimen = [10000,5000,4900]
#bsizes = [0.001,0.005,0.01,0.05,1]
bsizes = [0.01,0.05,0.1,0.5,1]

# Y-axis plot type (normal/relchange)
YPLOT = 'normal'

markers = ['o','v','^','<','>','s','p','P','H','D']
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot font sizes
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the y tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

for ifunc, func in enumerate(funcs):
    n = dimen[ifunc]

    # Set up plotting
    legend = []
    legend_lines = []

    blocks = [int(n*b+0.5) for b in bsizes]
    for ip, p in enumerate(blocks):

        # Set up plotting
        legend += ['Block Size '+str(p)]
        col = colours[ip]
        legend_lines += [Line2D([0],[0],color=col,linewidth=2)]

        # Load data
        Y = np.load(func+'_'+str(p)+'_plotdata.npy')
        Xt = np.load(func+'_'+str(p)+'_runtimes.npy')
        iters, insts = Y.shape # iterations, instances

        # Generate block size data
        Xb = np.cumsum([0]+[p]*(iters-1))
        Xi = np.arange(iters)

        # Calculate (relative) change if requested
        if YPLOT == 'relchange':
            Y = -np.diff(Y,axis=0)/Y[:-1,:] # (f_k-f_k+1)/f_k
            Xi = Xi[:-1]
            Xb = Xb[:-1]
            Xt = Xt[:-1,:]

        # Get no. runs (no randomness for GN)
        nruns = 1 if p == n else insts

        # Plot objective against iterations
        plt.figure(3*ifunc+1,figsize=(24,6))
        for iseed in range(nruns):
            plt.semilogy(Xi,Y[:,iseed],color=col,marker=markers[iseed],markevery=10)

        # Plot objective against runtime
        plt.figure(3*ifunc+2,figsize=(24,6))
        for iseed in range(nruns):
            plt.semilogy(Xt[:,iseed],Y[:,iseed],color=col,marker=markers[iseed],markevery=10)

        # Plot objective against block size
        plt.figure(3*ifunc+3,figsize=(24,6))
        for iseed in range(nruns):
            plt.semilogy(Xb,Y[:,iseed],color=col,marker=markers[iseed],markevery=10)

    # Label figures
    for fig in range(1,4):
        plt.figure(3*ifunc+fig,figsize=(24,6))
        plt.legend(legend_lines,legend,loc='upper right')
        if fig == 1:
            plt.xlabel('Iterations')
            subs = ''
        elif fig == 2:
            plt.xlabel('Runtime (s)')
            subs = '_time'
        elif fig==3:
            plt.xlabel('Cumulative block size')
            subs = '_blocks'
        if YPLOT == 'relchange':
            plt.ylabel('Relative Objective Change')
        else:
            plt.ylabel('Objective')
        plt.grid(True)
        plt.gcf().set_tight_layout(True)

        plt.savefig('/Users/jari/Desktop/'+func+subs+'.png')

    #plt.show()
