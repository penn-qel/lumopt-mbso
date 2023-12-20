from lumopt.optimization import Optimization
import os
import inspect

class MyOptimization(Optimization):
    """ See documentation in lumopt.optimization. Inherits same class with minor changes/improvements.
    For use as core optimization module that interfaces Lumerical, geometry, fom, and optimizer
    """

    def __init__(self, base_script, wavelengths, fom, geometry, optimizer, use_var_fdtd = False, hide_fdtd_cad = False, use_deps = True, plot_history = True, store_all_simulations = True, save_global_index = False, label=None, source_name = 'source', fields_on_cad_only = False):
        super().__init__(base_script, wavelengths, fom, geometry, optimizer, use_var_fdtd, hide_fdtd_cad, use_deps, plot_history, store_all_simulations, save_global_index, label, source_name, fields_on_cad_only)

        ## Figure out from which file this method was called (most likely the driver script)
        # Reruns this section so output doesn't end up in directory of this script
        frame = inspect.stack()[1]
        self.calling_file_name = os.path.abspath(frame[0].f_code.co_filename)
        self.base_file_path = os.path.dirname(self.calling_file_name)

    def process_forward_sim(self, iter, co_optimizations = None, one_forward = False):
        fom = super().process_forward_sim(iter, co_optimizations, one_forward)
        
        #Set this flag so that callable jac won't keep deciding to redo forward simulation because it thinks it
        #hasn't done it yet.
        if self.fields_on_cad_only:
            self.forward_fields = True

        return fom