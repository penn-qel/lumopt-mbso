###############################################################
# Class: KTransmissionFom

# Description: This class defines a FOM object based on power transmission through some region in spatial k-space, such as fitting within a certain fiber NA
# Author: Amelia Klein
###############################################################

from fom.transmissionfom import TransmissionFom
import numpy as np
import scipy.constants
import utils.ffthelpers as ffthelpers
from lumopt.figures_of_merit.modematch import ModeMatch

class KTransmissionFom(TransmissionFom):
    """Calculates FOM by integrating Poynting vector over a region in spatial k-space determined by boundary function

    Parameters
    -------------------
    :param monitor_name:        name of monitor that records FOM field

    Optional kwargs
    -------------------
    :kwarg NA               free-space NA to filter resulting fields into. Default 1

    Inherited kwargs
    -----------
    :kwarg direction:       direction of propagation ('Forward' or 'Backward') of the source mode. Default 'Forward'
    :kwarg multi_freq_src:  bool flag to enable / disable multi-frequency source calculation for adjoint. Default False
    :kwarg target_T_fwd:    function describing the target T_forward vs wavelength. Default lambda wl: np.ones(wl.size)
    :kwarg target_T_fwd_weights:    Takes in array of wavelength and returns weights for FOM integral. Default lambda wl: np.ones(wl.size)
    :kwarg boundary_func:   function of x,y,z arrays defining boundary for integral. Returns 1 if within region, 0 if outside. Default lambda x, y, z: np.ones(x.size)
    :kwarg norm_p:          exponent of the p-norm used to generate the FOM. Default 1
    :kwarg target_fom:      A target value for the FOM for printing/plotting distance of current design from target. Default 0
    :kwarg use_maxmin:      Boolean that triggers FOM/gradient calculations based on the worst-performing frequency, rather than average. Default False
    :kwarg prop_dist:       Positive distance to manually propagate fields from monitor to actual FOM plane. Default 0
    """

    def __init__(self, monitor_name, **kwargs):
        '''Initialization. See class docstring for list of kwargs'''

        super().__init__(monitor_name, **kwargs)

        NA = kwargs.get("NA", 1.0)
        self.NA = float(NA)

        print("Creating boundary function for NA of {}".format(self.NA))
        self.kboundary_func = ffthelpers.create_NA_boundary(self.NA)

        self.analysis_mode = False
        self.fields_saved = False


    def get_fom_fields(self, sim):
        '''Returns fields after applying appropriate filter in Fourier domain'''
        
        if not self.fields_saved:
            #Get fields from simulation monitor
            fields = super().get_fom_fields(sim)
            
            #Perform Fourier transform
            Ek, Hk, kx, ky, bufx, bufy = ffthelpers.getkfields(fields)
            
            #Save fields for future evaluations
            if self.analysis_mode:
                self.fields, self.Ek, self.Hk, self.kx, self.ky, self.bufx, self.bufy = fields, Ek, Hk, kx, ky, bufx, bufy
                self.fields_saved = True

            fields.E, fields.H = ffthelpers.filterkspace(fields, self.kboundary_func, Ek, Hk, kx, ky, bufx, bufy)
            return fields

        #Uses saved fields instead to skip copying from lumerical and performing forward fourier transform
        self.fields.E, self.fields.H = ffthelpers.filterkspace(self.fields, self.kboundary_func, self.Ek, self.Hk, self.kx, self.ky, self.bufx, self.bufy)
        return self.fields

    def enter_analysis(self):
        '''Sets flag to save computation time when repeatedly calculating ift in post-analysis'''
        self.analysis_mode = True

    def set_kboundary_func(self, func):
        self.kboundary_func = func