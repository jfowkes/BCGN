""" 20 CUTEst zero residual problems """

funcs = ['ARGTRIG','ARTIF','ARWHDNE','BDVALUES','BRATU2D','BRATU3D','BROWNALE','BROYDN3D','BROYDNBD','CBRATU2D',
             'CBRATU3D','CHANDHEQ','DRCAVTY1','DRCAVTY2','INTEGREQ','OSCIPANE','QR3D','QR3DBD','YATP1SQ','YATP2SQ']

args = [{'N':100},{'N':100},{'N':100},{'NDP':102},{'P':10},{'P':5},{'N':100},{'N':100},{'N':100},{'P':7},
            {'P':4},{'N':100},{'M':10},{'M':10},{'N':100},{'N':100},{'M':10},{'M':10},{'N':10},{'N':10}]
         
dimen = [100, 100, 100, 100, 64, 27, 100, 100, 100, 50, 16, 100, 100, 100, 100, 100, 155, 127, 120, 120]

nnzj = [10000, 300, 297, 300, 320, 189, 9910, 298, 684, 300, 128, 10000, 1300, 1300, 10000, 199, 2100, 1540, 500, 680]

#args = [{'N':200},{'N':500},{'N':500},{'NDP':502},{'P':22},{'P':8},{'N':200},{'N':500},{'N':500},{'P':16},
#            {'P':7},{'N':100},{'M':31},{'M':31},{'N':500},{'N':500},{'M':20},{'M':20},{'N':10},{'N':10}]

fxopts = [0]*20