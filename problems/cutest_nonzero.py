""" 19 CUTEst nonzero residual problems """
import numpy as np

funcs = ['ARGLALE','ARGLBLE','ARWHDNE','BRATU2DT','CHEMRCTB',
         'DRCAVTY2','FLOSP2HH','FLOSP2HL','FLOSP2HM','FREURONE',
         'PENLT1NE','PENLT2NE','LUKSAN12','LUKSAN13','LUKSAN14',
         'LUKSAN15','LUKSAN16','LUKSAN17','LUKSAN22']

args = [{'N':100},{'N':100},{'N':100},{'P':10},{'N':100},
        {'M':10},{'M':2},{'M':2},{'M':2},{'N':100},
        {'N':100},{'N':100},None,None,None,
        None,None,None,None]

dimen = [100,100,100,64,100,
         100,59,59,59,100,
         100,100,98,98,98,
         100,100,100,100]

# WARNING: these are all parameter dependent!!!
fxopts = 0.5*np.array([300,26600./267.,27.662029937166,0.0000185347366279219,0.0014044239811583,
                       0.00544960206525553,0.333333333333333,0.333333333333333,0.333333333333333,11964.5773486542,
                       9.0249999990975E-09,0.980937676348374,4292.19689140649,25188.8595896452,123.923540764503,
                       3.56969687722669,3.56969687722669,0.493161290322572,872.92295588937]) # since we assume 0.5||r(x)||^2
