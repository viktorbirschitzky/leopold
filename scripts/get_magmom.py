import numpy as np
import matplotlib.pyplot as plt
import pickle
from ase.io import read
import sys

if __name__ == '__main__':
    path, idx = sys.argv[1:]

    data = read(path+'/jump.xyz', index=idx)
    pol_state = data.arrays['pol_state']

    res = "MAGMOM = "

    for p in pol_state:
        res = res + str(p) + ' '
    res = res + '\n'
    print(res)

