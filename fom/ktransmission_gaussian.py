###############################################################
# Class: KTransmissionFom

# Description: This class defines a FOM object based on power transmission through some region in spatial k-space, such as fitting within a certain fiber NA
#               Uses Gaussian beam as adjoint source instead of rigorous result
# Author: Amelia Klein
###############################################################

from fom.ktransmissionfom import KTransmissionFom
from fom.transmissionfom import TransmissionFom
from lumopt.figures_of_merit.modematch import ModeMatch
import numpy as np
import scipy.constants

class KTransmissionFomGaussian(KTransmissionFom):
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
    :param polarization_angle:  (New in this FOM) polarization angle of Gaussian beam
    :param adj_scaling          Scaling factor of adjoint fields
    """

    def __init__(self, monitor_name, direction = 'Forward', boundary_func = None, NA = 1, 
                    multi_freq_src = False, target_T_fwd = lambda wl: np.ones(wl.size),
                    target_T_fwd_weights = lambda wl: np.ones(wl.size), 
                    source_precision = 10e-9, norm_p = 1, target_fom = 0, polarization_angle = 0, adj_scaling = 1):

        super().__init__(monitor_name, direction, boundary_func, NA, multi_freq_src, target_T_fwd, target_T_fwd_weights, source_precision, norm_p, target_fom)

        self.polarization_angle = polarization_angle
        self.adj_scaling = adj_scaling


    def initialize(self, sim):
        '''Initializes within optimization startup'''
        self.check_monitor_alignment(sim)
        self.wavelengths = ModeMatch.get_wavelengths(sim)
        TransmissionFom.add_index_monitor(sim, self.monitor_name, self.wavelengths.size)
        adjoint_injection_direction = 'Backward' if self.direction == 'Forward' else 'Forward'
        KTransmissionFomGaussian.add_adjoint_gaussian(sim, self.monitor_name, self.adjoint_source_name, adjoint_injection_direction, self.multi_freq_src)


    def make_adjoint_sim(self, sim):
        '''Adjont source is static so just has to enable it'''
        sim.fdtd.setnamed(self.adjoint_source_name, 'enabled', True)

    def get_adjoint_field_scaling(self, sim):
        omega = 2.0 * np.pi * scipy.constants.speed_of_light / self.wavelengths
        return 1j*omega*self.adj_scaling/(4*self.source_power)

    @staticmethod
    def add_adjoint_gaussian(sim, monitor_name, source_name, direction, multi_freq_source, polarization_angle = 0):
        '''Adds Gaussian beam adjoint source to simulation'''
        if sim.fdtd.getnamednumber('FDTD') == 1:
            sim.fdtd.addgaussian()
        else:
            raise UserWarning('No FDTD solver object could be found')
        sim.fdtd.set('name', source_name)
        sim.fdtd.select(source_name)
        monitor_type = sim.fdtd.getnamed(monitor_name, 'monitor type')
        geo_props, normal = ModeMatch.cross_section_monitor_props(monitor_type)
        sim.fdtd.setnamed(source_name, 'injection axis', normal.lower() + '-axis')
        for prop_name in geo_props:
            prop_val = sim.fdtd.getnamed(monitor_name, prop_name)
            sim.fdtd.setnamed(source_name, prop_name, prop_val)
        sim.fdtd.setnamed(source_name, 'override global source settings', False)
        sim.fdtd.setnamed(source_name, 'direction', direction)
        if sim.fdtd.haveproperty('multifrequency mode calculation'):
            sim.fdtd.setnamed(source_name, 'multifrequency field profile', multi_freq_source)
        #Sets polarization of beam in terms of angle in degrees. 0 degrees is p-polarized (x) and 90 is s-polarized (y)
        sim.fdtd.setnamed(source_name, 'polarization angle', polarization_angle)