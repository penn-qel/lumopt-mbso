###################################################################
# Class: constraints.py

# Description: This class generates constraint functions for a 3D
# metasurface of elliptical pillars allowed to move around within fabrication constraints
# Author: Amelia Klein
###################################################################

import numpy as np
from lumopt_mbso.geometry.constraints import PillarConstraints
from scipy.optimize import NonlinearConstraint
import itertools
from lumopt_mbso.utils.nearest_neighbor_iterator import nearest_neighbor_iterator
from scipy.sparse import csr_matrix
from collections import deque

class HexConstraints(PillarConstraints):
    """
        :param geo:             Handle to HexagonMetasurface object for optimization geometry
        :param manual_jac:      Boolean determining whether to provide manual jacobian calculation or to use finite differences
        :param print_warning:   Boolean determining whether to print when a constraint has a value < 0
        :param save_list:       Boolean determining whether to generate and save the list of constraints pairs once at start up
    """

    def __init__(self, geo, manual_jac = True, print_warning = True, save_list = False):
        '''The constructor for the HexConstraints class'''

        super().__init__(geo, manual_jac = manual_jac, print_warning = print_warning, save_list = save_list)

    def physical_constraint(self, params):
        '''Returns array of constraints according to pairs of elements in physical units of nm'''
        offset_x, offset_y, rad = self.geo.get_from_params(params)
        x = offset_x + self.geo.init_x
        y = offset_y + self.geo.init_y
        bound = self.geo.min_feature_size
        cons = []
        for pair in self.get_pair_iterator():
            i, j = pair[0], pair[1]
            cons.append((np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2) - rad[i] - rad[j] - bound))
        
        cons = np.array(cons)
        if self.print_warning and np.min(cons) < 0:
            print('Warning: Constraints violated by {} nm'.format(np.min(cons*1e9)))
        return cons

    def scaled_jacobian(self, params):
        '''Returns jacobian on same scale as the scaled constraint in sparse array format'''
        offset_x, offset_y, r = self.geo.get_from_params(params)
        x = offset_x + self.geo.init_x
        y = offset_y + self.geo.init_y
        N = x.size
        row,col,jac = [np.empty(self.num_constraints*6) for _ in range(3)]
        loc = 0
        row_number = 0
        for pair in self.get_pair_iterator():
            '''Calculates nonzero Jacobian elements. There should be 8 for each pair'''
            i, j = pair[0], pair[1]
            denom = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

            #Calculate positional components
            dxi = (x[i]-x[j])/denom
            dxj = (x[j]-x[i])/denom
            dyi = (y[i]-y[j])/denom
            dyj = (y[j]-y[i])/denom

            #Calculate radial components
            dri = -1
            drj = -1

            #Assign locations and values to relevant arrays
            row[loc],col[loc],jac[loc],loc=row_number,i,dxi,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,j,dxj,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,N+i,dyi,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,N+j,dyj,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,2*N+i,dri,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,2*N+j,drj,loc+1

            row_number += 1

        jac = csr_matrix((jac, (row,col)), shape=(self.num_constraints,3*N)).toarray()
        if hasattr(self.geo, "active"):
            return jac[:,np.concatenate((self.geo.active, self.geo.active, self.geo.active, self.geo.active, self.geo.active))]
        return jac
