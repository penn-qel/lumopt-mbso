###############################################################
# Class: KTransmissionFom

# Description: This class defines a FOM object based on power transmission through some region in spatial k-space, such as fitting within a certain fiber NA
# Author: Amelia Klein
###############################################################

from transmissionfom import TransmissionFom
import numpy as np
import scipy.constants
import ffthelpers
from lumopt.figures_of_merit.modematch import ModeMatch

class KTransmissionFom(TransmissionFom):
    """Calculates FOM by integrating Poynting vector over a region in spatial k-space determined by boundary function

    Parameters
    -------------------
    :param monitor_name:        name of monitor that records FOM field
    :param direction:           direction of propagation ('Forward' or 'Backward') of source
    :param multi_freq_src:      bool flag to enable/disable multi-frequency source calculation
    :param target_T_Fwd:        function describing target T_forward vs wavelength
    :param target_T_fwd_weights:function weighting different wavelengths for FOM
    :param boundary_func:       function defining boundary function for determining space to integrate over
    :param NA:                  Numerical aperture of target FOM. Used to auto-create a boundary function for this case
    :param norm_p:              exponent of p-norm used to generate FOM
    :param target_fom:          target value for FOM for printing/plotting distance of current design
    """

    def __init__(self, monitor_name, direction = 'Forward', boundary_func = None, NA = 1, 
                multi_freq_src = False, target_T_fwd = lambda wl: np.ones(wl.size),
                target_T_fwd_weights = lambda wl: np.ones(wl.size), 
                source_precision = 10e-9, norm_p = 1, target_fom = 0):

        super().__init__(monitor_name, direction = direction, multi_freq_src = multi_freq_src, target_T_fwd = target_T_fwd,
                        target_T_fwd_weights = target_T_fwd_weights, norm_p = norm_p, target_fom = target_fom)

        self.NA = float(NA)

        self.kboundary_func = boundary_func
        if self.kboundary_func is None:
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