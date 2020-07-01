""" Logistic Regression """
from __future__ import absolute_import, division, unicode_literals, print_function
from scipy.sparse import csr_matrix
from scipy.special import expit
import numpy as np

def logistic_regression(datasetname):
    """
    Logistic regression with regularisation as a nonlinear least squares problem:
    r_j(x) = phi_j(a_j^Tx) and r_m+1(x) = 0.5*lambda*||x||^2
    where
    phi_j(t) = ln(1+exp(-y_j*t)) for class labels y_j \in \{-1,1\}
    """
    LAMBDA = 1e-10 # regularisation

    # Load logistic regression dataset
    print('Loading logistic regression '+datasetname+' dataset...',end='',flush=True)

    if datasetname == 'gisette': # sparse arff format
        data = np.zeros((7000,5001))

        f = open('problems/'+datasetname+'.arff','r')
        found = False
        i = 0
        for line in f:
            if line.startswith('@data'):
                found = True
                continue
            if found:
                row = line.strip()[1:-1] # remove {}
                for entry in row.split(','):
                    j,v = entry.strip().split(' ')
                    data[i,int(j)] = float(v)
                i+=1
        f.close()

    else: # dense csv format
        data = np.loadtxt('problems/'+datasetname+'.csv',delimiter=',',skiprows=1)

    # Parse dataset
    a = data[:,:-1] # input data
    y = data[:,-1] # class labels
    if datasetname == 'chemotherapy':
        y = np.where(y==2,-1,y) # shift class labels from 1,2 to -1,1
    m,n = a.shape
    print('done')

    # Residual
    def r(x):
        res = np.zeros(m+1)
        for j in range(m):
            res[j] = np.logaddexp(0,-y[j]*np.dot(a[j,:],x))
        res[m] = 0.5*LAMBDA*np.dot(x,x) # regularisation term
        return res

    # Jacobian
    def J(x,indices=None):
        if indices is None:
            indices = np.arange(n)
        jac = np.zeros((m+1,len(indices)))

        for j in range(m):
            yaTx = y[j]*np.dot(a[j,:],x)

            # compute mirrored logistic term
            logistic = expit(-yaTx)

            # check if sparse
            if logistic == 0:
                continue
            else:
                jac[j,:] = -y[j]*a[j,indices]*logistic

        # regularisation term
        jac[m,:] = LAMBDA*x[indices]

        return csr_matrix(jac)
        #return jac

    # Initial guess
    x0 = np.zeros(n)

    return r, J, x0


# Test code
def main():
    import matplotlib.pyplot as plt

    #r, J, x0 = logistic_regression('chemotherapy')
    r, J, x0 = logistic_regression('gisette')
    #n = 61359
    n = 5001
    indices = np.random.permutation(n)[:100]

    # Get residual
    rx0 = r(x0)
    print(np.sum(rx0))

    # Get subset of Hessian
    Jx0 = J(x0,indices)

    # Plot Hessian sparsity
    plt.figure()
    plt.spy(np.dot(Jx0.T,Jx0),markersize=0.5)
    plt.ylabel('m')
    plt.xlabel('m')
    plt.title('Hessian Sparsity')
    plt.show()

if __name__ == "__main__":
    main()
