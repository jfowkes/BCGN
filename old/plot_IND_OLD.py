""" Individual function plots for paper """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

# Test functions
#funcs = ['chemotherapy','gisette']
#dimen = [61359,5000]
funcs = ['OSCIGRNE','ARTIF','BRATU2D']
dimen = [10000,5000,4900]
fxopts = [0,0,0]
bsizes = [0.01,0.05,0.1,0.5,1]
samp = sys.argv[1] # coordinate/gaussian/hashing

# Plot type (normal/relchange/normalised)
PLOT = 'normalised'
RESULTS = 'results/' # results folder
OUTPUT = 'figures/'

markers = ['o','v','^','<','>','s','p','P','H','D']
colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot font sizes
SMALL_SIZE = 12 #22
MEDIUM_SIZE = 14 #24
BIGGER_SIZE = 16 #26

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the y tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

for ifunc, func in enumerate(funcs):
    n = dimen[ifunc]
    fxopt = fxopts[ifunc]
    print(func)

    # Set up plotting
    legend = []
    legend_lines = []

    blocks = [int(n*b+0.5) for b in bsizes]
    if samp != 'coordinate':
        blocks += [2*n]
    for ip, p in enumerate(blocks):

        # Set up plotting
        if p == 2*n: # GN
            legend += ['GN']
            col = colours[ip-1] # purple
        elif p == n: # full block
            if samp == 'coordinate':
                legend += ['GN']
                col = colours[ip] # purple
            else:
                #legend += ['Full-Block']
                legend += ['RS-GN (1.0d)']
                col = colours[ip+1] # brown
        else: # other block sizes
            legend += ['RS-GN (' + str(bsizes[ip])+'d)']
            col = colours[ip]
        legend_lines += [Line2D([0],[0],color=col,linewidth=2)]

        # Load data
        if p == 2*n: # load GN
            Y = np.load(RESULTS+func+'_coordinate_'+str(n)+'_plotdata.npy')
            Xt = np.load(RESULTS+func+'_coordinate_'+str(n)+'_runtimes.npy')
        else: # other block sizes
            Y = np.load(RESULTS+func+'_'+samp+'_'+str(p)+'_plotdata.npy')
            Xt = np.load(RESULTS+func+'_'+samp+'_'+str(p)+'_runtimes.npy')
        iters, insts = Y.shape # iterations, instances
        Xi = np.arange(iters)

        # Generate block size data
        if p == 2*n: # GN
            Xb = np.cumsum([0]+[n]*(iters-1))
        else: # other block sizes
            Xb = np.cumsum([0]+[p]*(iters-1))

        # Calculate (relative) change if requested
        if PLOT == 'normalised':
            Y[Y == 0] = 1 # if starting from 0
            Y = (Y-fxopt)/(Y[0,:]-fxopt) # (f_k-f^*)/(f_0-f^*)
            Xb = Xb/n # units of n
        elif PLOT == 'relchange':
            Y = -np.diff(Y,axis=0)/Y[:-1,:] # (f_k-f_k+1)/f_k
            Xi = Xi[:-1]
            Xb = Xb[:-1]
            Xt = Xt[:-1,:]

        # Get no. runs (no randomness for GN)
        nruns = 1 if p == n or p == 2*n else insts

        # Plot objective against iterations
        plt.figure(3*ifunc+1)#,figsize=(24,6))
        for iseed in range(nruns):
            plt.semilogy(Xi,Y[:,iseed],color=col)#,marker=markers[iseed],markevery=10)

        # Plot objective against runtime
        plt.figure(3*ifunc+2)#,figsize=(24,6))
        for iseed in range(nruns):
            plt.semilogy(Xt[:,iseed],Y[:,iseed],color=col)#,marker=markers[iseed],markevery=10)

        # Plot objective against block size
        plt.figure(3*ifunc+3)#,figsize=(24,6))
        for iseed in range(nruns):
            plt.semilogy(Xb,Y[:,iseed],color=col)#,marker=markers[iseed],markevery=10)

    # Label figures
    for fig in range(1,4):
        plt.figure(3*ifunc+fig,figsize=(24,6))
        if fig == 1:
            plt.xlabel('Iterations')
            subs = 'iters'
        elif fig == 2:
            plt.xlabel('Runtime (s)')
            subs = 'runtime'
        elif fig==3:
            if PLOT == 'normalised':
                plt.xlabel('Cumulative block size (units of $d$)')
            else:
                plt.xlabel('Cumulative block size')
            subs = 'blocks'
            plt.legend(legend_lines,legend,loc='upper right')
        if PLOT == 'normalised':
            plt.ylabel('Normalised Objective')
        elif PLOT == 'relchange':
            plt.ylabel('Relative Objective Change')
        else:
            plt.ylabel('Objective')
        plt.grid(True)
        plt.title(func)
        plt.gcf().set_tight_layout(True)

        if OUTPUT:
            plt.savefig(OUTPUT+func+'_'+samp+'_'+subs+'.png')
            plt.close()

    if OUTPUT is None:
        plt.show()
