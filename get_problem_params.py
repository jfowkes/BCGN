import sys
sys.path.append('../pycutest/')
import pycutest

func = sys.argv[1]
pycutest.print_available_sif_params(func)

