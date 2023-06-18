###############################################################
# Class: KTransmissionCustomAdj

# Description: This class calculates FOM using KTransmissionFOM but internally uses CustomModeMatch for determining adjoint
# Author: Amelia Klein
###############################################################

from fom.ktransmissionfom import KTransmissionFom
from fom.transmissionfom import TransmissionFom
from fom.custommodematch import CustomModeMatch
import numpy as np


class KTransmissionCustomAdj(CustomModeMatch):
    """Calculates FOM as KTransmissionFOM but uses adjoint source as CustomModeMatch

    Parameters
    -------------------
    :param monitor_name:   name of the field monitor that records the fields to be used in the mode overlap calculation.
    :param Emodefun:       Function of x,y,z,wl (N,) arrays that returns a (N,3) vector describing the E field mode
    :param Hmodefun:       Function of x,y,z,wl (N,) arrays that returns a (N,3) vector describing the H field mode
    :param material:       Material that FOM is measured at as Material object or single epsilon

    Optional kwargs
    -------------------
    :kwarg direction:       direction of propagation ('Forward' or 'Backward') of the source mode. Default 'Forward'
    :kwarg NA               free-space NA to filter resulting fields into. Default 1
    :kwarg multi_freq_src:  bool flag to enable / disable multi-frequency source calculation for adjoint. Default False
    :kwarg target_T_fwd:    function describing the target T_forward vs wavelength. Default lambda wl: np.ones(wl.size)
    :kwarg target_T_fwd_weights:    Takes in array of wavelength and returns weights for FOM integral. Default lambda wl: np.ones(wl.size)
    :kwarg boundary_func:   function of x,y,z arrays defining boundary for integral. Returns 1 if within region, 0 if outside. Default lambda x, y, z: np.ones(x.size)
    :kwarg norm_p:          exponent of the p-norm used to generate the FOM. Default 1
    :kwarg target_fom:      A target value for the FOM for printing/plotting distance of current design from target. Default 0
    :kwarg use_maxmin:      Boolean that triggers FOM/gradient calculations based on the worst-performing frequency, rather than average. Default False
    :kwarg prop_dist:       Positive distance to manually propagate fields from monitor to actual FOM plane. Default 0
    """

    def __init__(self, monitor_name, Emodefun, Hmodefun, material, **kwargs):
        '''See class docstring for list of kwargs'''
        super().__init__(monitor_name, Emodefun, Hmodefun, material, **kwargs)

        self.ktransfom = KTransmissionFom(monitor_name, **kwargs)

    def initialize(self, sim):
        '''Initializes within optimization startup'''

        #Run Custom Mode Match initialize
        super().initialize(sim)

        #Add index monitor so can get eps in fields
        TransmissionFom.add_index_monitor(sim, self.monitor_name, self.wavelengths.size)
        self.ktransfom.wavelengths = self.wavelengths

    def get_fom(self, sim):
        '''returns FOM and saves intermediate data'''

        #Runs Custom Mode Match FOM so that certain scaling factors are calculated/saved
        cmmfom = super().get_fom(sim)
        print("Mode overlap FOM: {}".format(cmmfom))

        #Returns KTransmissionFOM
        return self.ktransfom.get_fom(sim)

    def get_adjoint_field_scaling(self, sim):
        '''Returns adjoint scaling as if it were CMM FOM'''
        return super().get_adjoint_field_scaling(sim)

    def enter_analysis(self):
        '''Sets flag to save computation time when repeatedly calculating ift in post-analysis'''
        self.ktransfom.enter_analysis()