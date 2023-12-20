###################################################################
# Class: constraints.py

# Description: This class generates constraint functions for a 3D
# metasurface of elliptical pillars allowed to move around within fabrication constraints
# Author: Amelia Klein
###################################################################

import numpy as np
from scipy.optimize import NonlinearConstraint
import itertools
from utils.nearest_neighbor_iterator import nearest_neighbor_iterator
from scipy.sparse import csr_matrix
from collections import deque

class PillarConstraints(object):
    """
        :param geo:             Handle to MovingMetasurface3D object for optimization geometry
        :param radius_type:     Flag determining if calculating mean or max of ellipses' two radii. Valid inputs 'mean' or 'max'
        :param manual_jac:      Boolean determining whether to provide manual jacobian calculation or to use finite differences
        :param print_warning:   Boolean determining whether to print when a constraint has a value < 0
        :param save_list:       Boolean determining whether to generate and save the list of constraints pairs once at start up
    """

    def __init__(self, geo, radius_type = 'mean', manual_jac = True, print_warning = True, save_list = False):
        '''The constructor for the PillarConstraints class'''

        self.geo = geo
        self.num_pillars = geo.offset_x.size
        self.print_warning = print_warning
        self.radius_type = radius_type
        self.manual_jac = True
        if not (self.radius_type == 'mean' or self.radius_type == 'max'):
            raise UserWarning("Valid radius types are 'mean' and 'max'")
        self.save_list = save_list
        self.constraint_list = None
        self.num_constraints = sum(1 for _ in self.get_pair_iterator())
    
    def get_pair_iterator(self):
        '''Returns saved list of pairs if it exists or calls generator otherwise'''
        #Returns iterable list if it exists
        if self.constraint_list is not None:
            return self.constraint_list

        #Generates list on first run through
        if self.save_list:
            print("Generating constraints")
            self.constraint_list = deque(self.get_pair_iterator_impl())
            print("Constrained pairs saved")
            return self.constraint_list

        return self.get_pair_iterator_impl()

    def get_pair_iterator_impl(self):
        '''Creates iterator that generates all possible combinations of pillars. Optionally limited to nearest neighbors.'''
        if self.geo.limit_nearest_neighbor_cons:
            nx = self.geo.grid_shape[0]
            ny = self.geo.grid_shape[1]
            return nearest_neighbor_iterator(np.arange(self.num_pillars), nx, ny)
        else:
            return itertools.combinations(np.arange(self.num_pillars), 2)

    def physical_constraint(self, params):
        '''Returns array of constraints according to pairs of elements in physical units of nm'''
        offset_x, offset_y, rx, ry, phi = self.geo.get_from_params(params)
        x = offset_x + self.geo.init_x
        y = offset_y + self.geo.init_y
        if self.radius_type == 'max':
            rad = np.maximum(rx, ry)
        else:
            rad = (rx + ry) / 2
        bound = self.geo.min_feature_size
        cons = []
        for pair in self.get_pair_iterator():
            i, j = pair[0], pair[1]
            cons.append((np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2) - rad[i] - rad[j] - bound))
        
        cons = np.array(cons)
        if self.print_warning and np.min(cons) < 0:
            print('Warning: Constraints violated by {} nm'.format(np.min(cons*1e9)))
        return cons

    def scaled_constraint(self, params):
        '''Returns constraint scaled by geometry scaling factor to be used in optimization algorithm'''
        return self.physical_constraint(params)*self.geo.scaling_factor

    def scaled_jacobian(self, params):
        '''Returns jacobian on same scale as the scaled constraint in sparse array format'''
        offset_x, offset_y, rx, ry, phi = self.geo.get_from_params(params)
        x = offset_x + self.geo.init_x
        y = offset_y + self.geo.init_y
        N = x.size
        row,col,jac = [np.empty(self.num_constraints*8) for _ in range(3)]
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
            if self.radius_type == 'max':
                drxi = -1 if rx[i] > ry[i] else 0
                drxj = -1 if rx[j] > ry[j] else 0
                dryi = -1 if ry[i] > rx[i] else 0
                dryj = -1 if ry[j] > rx[j] else 0
            else:
                drxi = -0.5
                drxj = -0.5
                dryi = -0.5
                dryj = -0.5

            #Assign locations and values to relevant arrays
            row[loc],col[loc],jac[loc],loc=row_number,i,dxi,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,j,dxj,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,N+i,dyi,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,N+j,dyj,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,2*N+i,drxi,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,2*N+j,drxj,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,3*N+i,dryi,loc+1
            row[loc],col[loc],jac[loc],loc=row_number,3*N+j,dryj,loc+1

            row_number += 1

        jac = csr_matrix((jac, (row,col)), shape=(self.num_constraints,5*N)).toarray()
        if hasattr(self.geo, "active"):
            return jac[:,np.concatenate((self.geo.active, self.geo.active, self.geo.active, self.geo.active, self.geo.active))]
        return jac

    def physical_jacobian(self, params):
        '''Returns jacobian relative to the actual parameter values'''
        return self.scaled_jacobian(params)/self.geo.scaling_factor

    def get_constraint_dict(self):
        '''Returns dict defining constraints to be used in SLSQP optimization'''
        if self.manual_jac:
            return {'type': 'ineq', 'fun': self.scaled_constraint, 'jac': self.scaled_jacobian}
        else:
            return {'type': 'ineq', 'fun': self.scaled_constraint}           

    def identify_violated_constraints(self, params, tol = 0):
        '''Identifies by constraint index which constraints have been violated within tol'''
        cons = self.physical_constraint(params)
        locations = np.nonzero((cons - tol) < 0.0)
        return locations[0]

    def count_violated_constraints(self, params, tol = 0):
        '''Counts number of violated constraints within tolerance'''
        locations = self.identify_violated_constraints(params, tol)
        return locations.size

    def identify_constrained_pillars(self, params, tol = 0):
        '''Returns set of pillar indices which have at least one constrained parameter'''
        constraint_inds = self.identify_violated_constraints(params, tol)
        constrained_pillars = set()
        for i, pair in enumerate(self.get_pair_iterator()):
            if np.isin(i, constraint_inds):
                constrained_pillars.update(pair)
        return constrained_pillars

    def disable_warnings(self):
        '''Turns off functionality to print violation warnings'''
        self.print_warning = False
