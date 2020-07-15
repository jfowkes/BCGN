""" Individual function plots for SGN """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Test functions
funcs = ['DMN15103','SPMSQRT','ARWHDNE','FREURONE']
dimen = [99,4999,5000,5000]
samps = [4643,8329,9998,9998]
sketches = ['No Sketching','Coordinate','1-Hashing','2-Hashing']
b = 1.2

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

    for isk, sk in enumerate(sketches):
        
        # Handle GN seperately
        if sk == 'No Sketching':
            p = samps[ifunc]
            subs = ''
        else:
            p = int(n*b+0.5)
            subs = '_'+sk.upper()

        # Set up plotting
        legend += [sk]
        col = colours[isk]
        legend_lines += [Line2D([0],[0],color=col,linewidth=2)]

        # Load data
        Y = np.load(func+'_'+str(p)+subs+'_plotdata.npy')
        Xt = np.load(func+'_'+str(p)+subs+'_runtimes.npy')
        iters, insts = Y.shape # iterations, instances
        Xi = np.arange(iters)

        # Generate block size data
        #Xb = np.cumsum([0]+[p]*(iters-1))

        # Get no. runs (no randomness for GN)
        nruns = 1 if sk == 'No Sketching' else insts

        # Plot f_k against iterations
        plt.figure(2*ifunc+1,figsize=(24,6))
        for iseed in range(nruns):
            plt.semilogy(Xi,Y[:,iseed],color=col,marker=markers[iseed],markevery=10)
            
        # Plot f_k against runtime
        plt.figure(2*ifunc+2,figsize=(24,6))
        for iseed in range(nruns):
            plt.semilogy(Xt[:,iseed],Y[:,iseed],color=col,marker=markers[iseed],markevery=10)

    # Label figures
    for fig in range(1,3):
        plt.figure(2*ifunc+fig,figsize=(24,6))
        plt.legend(legend_lines,legend,loc='upper right')
        if fig == 1:
            plt.xlabel('Iterations')
            subs = ''
        elif fig == 2:
            plt.xlabel('Runtime (s)')
            subs = '_time'
        plt.ylabel('Objective')
        plt.grid(True)
        plt.gcf().set_tight_layout(True)

        plt.savefig('/home/jari/Desktop/'+func+subs+'.png')

    #plt.show()
