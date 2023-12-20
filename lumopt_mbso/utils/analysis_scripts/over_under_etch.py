'''Takes in command line a path to simulation file and path for final dataset.
Performs simulation on a set of (dr) radius perturbations. Saves result
matrix into output file

   USAGE: over_under_etch.py [sim_file_path] [savedata_path]'''

import numpy as np
import sys
import scipy
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.wavelengths import Wavelengths
from geometry.moving_metasurface3D import MovingMetasurface3D
from fom.ktransmissionfom import KTransmissionFom
from utils.chebyshev import chebyshev
from utils.spectrum_weights import get_spectrum_weights

def calculate_perturbed_radius(sim, surface, fom, dr):
   '''Run simulation with radii perturbed by amount dr'''

   #Get starting parameters
   params = surface.get_current_params()
   offset_x, offset_y, rx, ry, phi = surface.get_from_params(params)

   #Perturb pillar radii
   new_params = surface.get_scaled_params(offset_x, offset_y, np.maximum(rx + dr, 0), np.maximum(ry + dr, 0), phi)

   #Add modified parameters into geometry
   surface.add_geo(sim, new_params, only_update = True)

   #Run simulation
   print("Running simulation for perturbed radius dr = {:.0f} nm".format(dr*1e9))
   sim.fdtd.run()

   #Get FOM
   T = fom.get_fom(sim)

   #Reset geometry
   surface.add_geo(sim, params, only_update = True)

   return T

def main(args):
   sourcepath = args[0]
   destpath = args[1]

   #Pull geometry and wavelengths from simulation
   surface, wavelengths = MovingMetasurface3D.create_from_existing_simulation(sourcepath, get_wavelengths = True)
   spectrumdata = scipy.io.loadmat('../datasets/NVspectrum')
   weights = get_spectrum_weights(wavelengths, spectrumdata['wl'].flatten()*1e-9, spectrumdata['spectrum'].flatten())

   #Open simulation
   sim = Simulation('./', use_var_fdtd = False, hide_fdtd_cad = True)
   sim.load(sourcepath)

   #Store initial geometry parameters
   init_params = surface.get_current_params()

   #Array of radius perturbations in nm to measure (start/stop/step)
   dr_range = np.linspace(-50, 50, 101)

   #FOM object
   fom = KTransmissionFom('fom', NA = 0.2, target_T_fwd_weights = scipy.interpolate.interp1d(wavelengths, weights, kind='nearest', fill_value='extrapolate'))
   fom.wavelengths = wavelengths
   Tmatrix = np.array([])

   #Perform simulations
   for dr in dr_range:
      T = calculate_perturbed_radius(sim, surface, fom, dr*1e-9)
      Tmatrix = np.append(Tmatrix, T)

   #Save results
   np.savez(destpath, Tmatrix=Tmatrix, dr=dr_range)

   sim.remove_data_and_save()

if __name__ == "__main__":
   main(sys.argv[1:])