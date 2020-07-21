import sys
sys.path.append('../pycutest/')
import pycutest

#from problems.cutest32_zero import funcs, args
#from problems.cutest_nonzero import funcs, args
funcs = ['OSCIGRNE','ARTIF','BRATU2D']
args = [{'N':10000},{'N':5000},{'P':72}]

for ifunc, func in enumerate(funcs):
    print('=================')
    print(func)
    prob = pycutest.import_problem(func,sifParams=args[ifunc])
    print('n: ',end='')
    print(prob.n)
    print('m: ',end='')
    print(prob.m)
print('=================')
