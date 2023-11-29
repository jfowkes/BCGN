"""
28 CUTEst large problems (dimen ~ 1000)

From Appendix B of C .Cartis and L. Roberts (but with dimen ~ 1000):
Scalable Subspace Methods for Derivative-Free Nonlinear Least-Squares Optimization
"""

funcs = ['ARGLALE','ARGLBLE','ARGTRIG','ARTIF','ARWHDNE',
         'BDVALUES','BRATU2D','BRATU2DT','BRATU3D','BROWNALE',
         'BROYDN3D','BROYDNBD','CBRATU2D','CHANDHEQ','EIGENB',
         'FREURONE','INTEGREQ','MOREBVNE','MSQRTA','MSQRTB',
         'OSCIGRNE','PENLT1NE','POWELLSE','SEMICN2U','SPMSQRT',
         'VARDIMNE','YATP1SQ','YATP2SQ']

args = [{'N':1000,'M':2000},{'N':1000,'M':2000},{'N':1000},{'N':1000},{'N':1000},
        {'NDP':1002},{'P':32},{'P':32},{'P':12},{'N':1000},
        {'N':1000},{'N':1000},{'P':23},{'N':1000},{'N':50},
        {'N':1000},{'N':1000},{'N':1000},{'P':32},{'P':32},
        {'N':1000},{'N':1000},{'N':1000},{'N':1000,'LN':900},{'M':334},
        {'N':1000},{'N':50},{'N':50}]

# as found by get_test_set_info.py
# (excludes fixed variables)
dimen = [1000,1000,1000,1000,1000,
         1000,900,900,1000,1000,
         1000,1000,882,1000,2550,
         1000,1000,1000,1024,1024,
         1000,1000,1000,1000,1000,
         1000,2600,2600]

# as found by get_scipy_min.py
# WARNING: these are all parameter (args) dependent!!!
fxopts = [500,249.8125446103489,0,0,139.5675146829764,
          0,0,5.2862467114649865e-09,0,0,
          0,0,0,0,0,
          60734.85505474419,0,0,0,0,
          0,4.843136116592137e-08,0,0,0,
          0,0,0]
