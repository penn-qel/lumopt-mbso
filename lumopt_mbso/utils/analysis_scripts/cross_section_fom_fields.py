'''Runs simulation to obtain cross-section and fom fields, and save data to file

   USAGE: cross_section_fom_fields.py [sim_file_path] [savedata_path]'''

import numpy as np
import sys
import scipy
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.lumerical_methods.lumerical_scripts import get_fields
from lumopt.figures_of_merit.modematch import ModeMatch

def main(args):
   sourcepath = args[0]
   destpath = args[1]

   #Simulate for two dipole orientations
   theta1 = 90
   theta2 = 35.3
   phi1 = 0
   phi2 = 90

   #Open simulation
   sim = Simulation('./', use_var_fdtd=False, hide_fdtd_cad = True)
   sim.load(sourcepath)
   sim.fdtd.switchtolayout()
   sim.fdtd.select('source')
   sim.fdtd.set('theta', theta1)
   sim.fdtd.set('phi', phi1)

   print("Running simulation for dipole theta = {:.1f}, phi = {:.1f}".format(theta1, phi1))
   sim.fdtd.run()

   cross_section1 = get_fields(sim.fdtd, 'cross_section', 'cross_section_fields1', get_eps = False, get_D = False, get_H = False, nointerpolation = False)
   fom1 = get_fields(sim.fdtd, 'fom', 'fom_fields1', get_eps = False, get_D = False, get_H = False, nointerpolation = False)
   sp1 = ModeMatch.get_source_power(sim, fom1.wl)

   sim.fdtd.switchtolayout()
   sim.fdtd.select('source')
   sim.fdtd.set('theta', theta2)
   sim.fdtd.set('phi', phi2)

   print("Running simulation for dipole theta = {:.1f}, phi = {:.1f}".format(theta2, phi2))
   sim.fdtd.run()

   cross_section2 = get_fields(sim.fdtd, 'cross_section', 'cross_section_fields2', get_eps = False, get_D = False, get_H = False, nointerpolation = False)
   fom2 = get_fields(sim.fdtd, 'fom', 'fom_fields2', get_eps = False, get_D = False, get_H = False, nointerpolation = False)
   sp2 = ModeMatch.get_source_power(sim, fom2.wl)

   #Clear Lumerical simulation
   sim.fdtd.switchtolayout()
   sim.remove_data_and_save()

   #Save data
   np.savez(destpath + '_cross_section', x1= cross_section1.x, y1 = cross_section1.y, z1 = cross_section1.z, wl1 = cross_section1.wl, E1 = cross_section1.E,
   		x2 = cross_section2.x, y2 = cross_section2.y, z2 = cross_section2.z, wl2 = cross_section2.wl, E2 = cross_section2.E, theta1 = theta1, theta2 = theta2, phi1 = phi1, phi2 = phi2, sp1 = sp1, sp2 = sp2)
   np.savez(destpath + '_fom', x1= fom1.x, y1 = fom1.y, z1 = fom1.z, wl1 = fom1.wl, E1 = fom1.E,
   		x2 = fom2.x, y2 = fom2.y, z2 = fom2.z, wl2 = fom2.wl, E2 = fom2.E, theta1 = theta1, theta2 = theta2, phi1 = phi1, phi2 = phi2, sp1 = sp1, sp2 = sp2)

if __name__ == "__main__":
   main(sys.argv[1:])