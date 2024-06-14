# gives all distinct hopping path ways occuring from a set of hopping vectors
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('hops.pkl', 'rb') as f:
    idxs, hops = pickle.load(f)

hops = np.abs(hops)

"""
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.hist(hops[:,i], bins=100)
    plt.yscale('log')
plt.show()

"""
hops = np.round(hops)
"""
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.hist(hops[:,i], bins=100)
    plt.yscale('log')
plt.show()
"""
hopping_mechs, counts = np.unique(hops, axis=0, return_counts=True)

comp_mat = np.all(hopping_mechs[:,np.newaxis] == hopping_mechs[np.newaxis:,[1,0,2]], axis=-1)

new_counts = []
new_hops = []
for i,j in zip(*np.triu(comp_mat).nonzero()):
    if i==j:
        new_counts.append(counts[i])
        new_hops.append(hopping_mechs[i])
    else:
        new_counts.append(counts[i]+counts[j])
        new_hops.append(hopping_mechs[i])

for h,c in zip(new_hops, new_counts):
    print(f"Hop along {h} occured {c} times {np.round(c*100/np.sum(new_counts), 2)}")
