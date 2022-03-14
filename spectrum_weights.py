import numpy as np
import scipy as np

'''Calculates relevant weights for desired simwl using data from spectrumwl and counts'''
def get_spectrum_weights(simwl, spectrumwl, counts):
    #Assume wavelengths are evenly space
    diff = simwl[1] - simwl[0]
    bin_edges = np.append(simwl - diff/2, simwl[len(simwl) - 1] + diff/2)

    locations = np.searchsorted(spectrumwl, bin_edges)

    weights = np.zeros(len(simwl))
    for i in range(len(simwl)):
        weights[i] = np.trapz(y=counts[locations[i]:locations[i+1]], x=spectrumwl[locations[i]:locations[i+1]])
            
    #Return normalized weights
    return weights / np.trapz(y=counts, x = spectrumwl)