###############################################################
# Class: KTransmissionFom

# Description: This class defines a FOM object based on power transmission through some region in spatial k-space, such as fitting within a certain fiber NA
# Author: Amelia Klein
###############################################################

from transmissionfom import TransmissionFom
import numpy as np
import scipy.constants
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
            self.kboundary_func = KTransmissionFom.create_NA_boundary(self.NA)


    def get_fom_fields(self, sim):
        #Get fields from monitor as in parent class
        monitor_fields = super().get_fom_fields(sim)

        #Filter fields in Fourier space before returning
        return KTransmissionFom.filterkspace(monitor_fields, self.kboundary_func)
        
    @staticmethod
    def filterkspace(fields, kboundary_func):
        #Applies boundary function in frequency space on fields object, returns it

        assert(fields.z.size == 1)
        Ek, vx, vy = KTransmissionFom.fft2D(fields.E, fields.x, fields.y)
        Hk, vx, vy = KTransmissionFom.fft2D(fields.H, fields.x, fields.y)

        #Gets grid of relevant points. Normalizes spatial frequencies by multiplying by wavelength and gets boundary weights
        vxv, vyv, wlv = np.meshgrid(vx, vy, fields.wl, indexing = 'ij')
        kx = vxv * wlv
        ky = vyv * wlv
        weights = kboundary_func(kx.flatten(), ky.flatten()).reshape(fields.x.size, fields.y.size, 1, fields.wl.size, 1)

        fields.E = np.fft.ifftn(Ek*weights, axes=(0,1))
        fields.H = np.fft.ifftn(Hk*weights, axes=(0,1))

        return fields
       
    @staticmethod
    def fft2D(A, x, y):
        #Calculates fft over first two axes of our 5D field arrays. Returns new field and frequencies

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        nx = x.size
        ny = y.size

        #Performs fft
        Ak = np.fft.fftn(A, axes=(0,1))

        #Gets spatial frequencies
        vx = np.fft.fftfreq(nx, dx)
        vy = np.fft.fftfreq(ny, dy)

        return Ak, vx, vy 

        
    @staticmethod
    def create_NA_boundary(NA):
        
        #Takes in as inputs normalized kx and ky. Returns 1 if within spot and 0 otherwise
        def boundary(kx, ky):
            return np.square(kx) + np.square(ky) < NA**2

        return boundary
