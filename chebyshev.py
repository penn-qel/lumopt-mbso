import numpy as np
import scipy.constants

#Returns a wavelength array that is spread on a chebyshev grid in frequency
def chebyshev(minwl, maxwl, N):
	minf = scipy.constants.speed_of_light / maxwl
	maxf = scipy.constants.speed_of_light / minwl

	#Creates chebyshev grid in interval [-1,1]
	points = np.cos((np.arange(N)+0.5)*np.pi/N)

	#Scales so min/max are represented
	scale = (maxf - minf)/2
	points = scale*points

	#Shifts center to midpoint of frequencies
	center = (minf + maxf)/2
	freqs = points + center

	#Converts back to wavelength and returns
	return scipy.constants.speed_of_light / freqs