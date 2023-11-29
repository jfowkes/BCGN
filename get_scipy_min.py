from scipy.optimize import least_squares
import numpy as np
import pycutest

# Test functions
funcs = ['ARGLALE','ARGLBLE','ARWHDNE','BRATU2DT','FREURONE','PENLT1NE']
args = [{'N':1000,'M':2000},{'N':1000,'M':2000},{'N':1000},{'P':32},{'N':1000},{'N':1000}]

# Loop over test functions
for ifunc, func in enumerate(funcs):
    print('====== '+func+' ======')

    # Load pycutest problem
    prob = pycutest.import_problem(func,sifParams=args[ifunc])
    def r(x): return prob.cons(x)
    def J(x): return prob.scons(x,gradient=True)[1].tocsr()
    x0 = prob.x0

    # Find minimum
    eps = np.finfo(float).eps # machine precision
    res = least_squares(r, x0, J, ftol=eps, xtol=eps, gtol=eps)
    print(res.cost)
