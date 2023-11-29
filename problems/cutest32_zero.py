""" 32 CUTEst zero residual problems """

funcs = ['ARGTRIG','ARTIF','BDVALUES','BRATU2D','BROWNALE','BROYDN3D','BROYDNBD','CBRATU2D','CHANDHEQ','CHEMRCTA',
         'CHNRSBNE','DRCAVTY1','DRCAVTY3','EIGENA','EIGENB','FLOSP2TL','FLOSP2TM','HYDCAR20','INTEGREQ','MOREBVNE',
         'MSQRTA','MSQRTB','OSCIGRNE','POWELLSE','SEMICN2U','SEMICON2','SPMSQRT','VARDIMNE','LUKSAN11','LUKSAN21',
         'YATP1NE','YATP2SQ']

args = [{'N':100},{'N':100},{'NDP':102},{'P':10},{'N':100},{'N':100},{'N':100},{'P':7},{'N':100},{'N':50},
        {'N':50},{'M':10},{'M':10},{'N':10},{'N':10},{'M':2},{'M':2}, None,{'N':100},{'N':100},
        {'P':10},{'P':10},{'N':100},{'N':100},{'N':100,'LN':90},{'N':100,'LN':90},{'M':34},{'N':100},None,None,
        {'N':10},{'N':10}]

dimen = [100, 100, 100, 64, 100, 100, 100, 50, 100, 100,
         50, 100, 100, 110, 110, 59, 59, 99, 100, 100,
         100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
         120, 120]

fxopts = [0]*32
