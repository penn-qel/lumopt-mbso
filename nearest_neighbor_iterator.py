import numpy as np
import itertools

'''Yields a tuple of combinations of parameters, limited to nearest-neighbors on a nx x ny grid'''
def nearest_neighbor_iterator(iterable, nx, ny):
	#Take standard combination of elements
	pool = tuple(iterable)
	n = len(pool)
	index_id = np.arange(n).reshape(nx, ny)
	for index_pair in itertools.combinations(iterable, 2):
		index1 = index_pair[0]
		index2 = index_pair[1]
		i1 = index1 // ny
		i2 = index2 // ny
		j1 = index1 % ny
		j2 = index2 % ny
		#Exclusive or that indices are off by one (adjacent)
		if (i2 - i1 == 1 and j1 == j2) or (i1 == i2 and j2 - j1 == 1):
			yield tuple(pool[i] for i in index_pair)