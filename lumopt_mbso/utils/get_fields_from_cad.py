import lumapi
from lumopt.utilities.fields import Fields, FieldsNoInterp
import numpy as np
import scipy as sp

def get_fields_from_cad(fdtd, field_result_name, get_eps, get_D, get_H, nointerpolation, clear_result = True):
    '''Pulls field object saved into CAD workspace under name field_result_name. Optionally deletes
    variable on CAD side after pulling into python to reduce duplicate instances in memory.
    Mostly copies the get_fields function without the step of putting the data into CAD workspace first'''
    #Pull field dict
    fields_dict = lumapi.getVar(fdtd.handle, field_result_name)
    #Delete item from CAD after pulling to save memory
    if clear_result:
        fdtd.eval("clear({});".format(field_result_name))

    if get_eps:
        if fdtd.getnamednumber('varFDTD') == 1:
            if 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and not 'index_z' in fields_dict['index']: # varFDTD TE simulation
                fields_dict['index']['index_z'] = fields_dict['index']['index_x']*0.0 + 1.0
            elif not 'index_x' in fields_dict['index'] and not 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']: # varFDTD TM simulation
                fields_dict['index']['index_x'] = fields_dict['index']['index_z']*0.0 + 1.0
                fields_dict['index']['index_y'] = fields_dict['index']['index_x']
        assert 'index_x' in fields_dict['index'] and 'index_y' in fields_dict['index'] and 'index_z' in fields_dict['index']
        fields_eps = np.stack((np.power(fields_dict['index']['index_x'], 2), 
                               np.power(fields_dict['index']['index_y'], 2), 
                               np.power(fields_dict['index']['index_z'], 2)), 
                               axis = -1)
    else:
        fields_eps = None

    fields_D = fields_dict['E']['E'] * fields_eps * sp.constants.epsilon_0 if get_D else None

    fields_H = fields_dict['H']['H'] if get_H else None

    if nointerpolation:
        deltas = [fields_dict['delta']['x'], fields_dict['delta']['y'], fields_dict['delta']['z']]
        return FieldsNoInterp(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], deltas, fields_dict['E']['E'], fields_D, fields_eps, fields_H)
    else:
        return Fields(fields_dict['E']['x'], fields_dict['E']['y'], fields_dict['E']['z'], fields_dict['E']['lambda'], fields_dict['E']['E'], fields_D, fields_eps, fields_H)