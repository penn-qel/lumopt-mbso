'''Takes in command line a path to simulation file and path for final dataset.
Performs simulation on a set of sidewall angles.

   USAGE: sidewall_tolerance.py [sim_file_path] [savedata_path]'''

import numpy as np
import sys
import scipy
import os
from lumopt.utilities.simulation import Simulation
from lumopt.utilities.wavelengths import Wavelengths
from geometry.moving_metasurface3D import MovingMetasurface3D
from geometry.moving_metasurface3D_sidewall import MovingMetasurface3DSidewall
from fom.ktransmissionfom import KTransmissionFom
from utils.chebyshev import chebyshev
from utils.spectrum_weights import get_spectrum_weights

def calculate_sidewall(sim, params, fom, theta):
   #Define geometry
   num_chunks = 31

   geom = MovingMetasurface3DSidewall(posx = params['posx'], posy = params['posy'], rx = params['rx'], ry = params['ry'], 
            min_feature_size = 0, z = params['z'], h = params['h'], eps_in = 2.4**2, eps_out = 1, 
            phi = params['phi'], scaling_factor = 1, phi_scaling = 1, limit_nearest_neighbor_cons = False,
            height_precision = num_chunks, sidewall_angle = theta)

   #Add geometry into simulation
   geom.add_geo(sim, None, False)

   #Run simulation
   print("Running simulation for sidewall angle theta = {:.1f}".format(theta))
   sim.fdtd.run()

   #Get FOM
   T = fom.get_fom(sim)

   #Remove pillar objects from simulation
   sim.fdtd.switchtolayout()
   for i in np.arange(num_chunks):
      sim.fdtd.select('Pillars' + str(i))
      sim.fdtd.delete()

   #Return FOM
   return T

def main(args):
   sourcepath = args[0]
   destpath = args[1]

   #Pull geometry and wavelengths from simulation
   params = MovingMetasurface3D.get_params_from_existing_simulation(sourcepath, get_wavelengths = True)
   wavelengths = params['wl']
   spectrumdata = scipy.io.loadmat('../datasets/NVspectrum')
   weights = get_spectrum_weights(wavelengths, spectrumdata['wl'].flatten()*1e-9, spectrumdata['spectrum'].flatten())

   #Open simulation
   sim = Simulation('./', use_var_fdtd = False, hide_fdtd_cad = True)
   sim.load(sourcepath)

   #Copy simulation file
   sim.save('sidewall_tolerance_simulation_temp')

   #Delete pillars object
   sim.fdtd.select('Pillars')
   sim.fdtd.delete()

   #Define sidewall angles to test
   thetav = np.linspace(80,110,151)
   Tmatrix = np.array([])

   #FOM object
   fom = KTransmissionFom('fom', NA = 0.2, target_T_fwd_weights = scipy.interpolate.interp1d(wavelengths, weights, kind='nearest', fill_value='extrapolate'))
   fom.wavelengths = wavelengths

   #Perform simulations
   for theta in thetav:
      T = calculate_sidewall(sim, params, fom, theta)
      Tmatrix = np.append(Tmatrix, T)

   #delete temp simulation file
   sim.remove_data_and_save()
   os.remove("sidewall_tolerance_simulation_temp.fsp")
   os.remove("sidewall_tolerance_simulation_temp_p0.log")

   #Save results
   np.savez(destpath, Tmatrix=Tmatrix, theta=thetav)

if __name__ == "__main__":
   main(sys.argv[1:])