###################################################################
# Class: constraints.py

# Description: This class generates constraint functions for a 3D
# metasurface of elliptical pillars allowed to move around within fabrication constraints
# Author: Amelia Klein
###################################################################

import numpy as np
import itertools
from nearest_neighbor_iterator import nearest_neighbor_iterator

class PillarConstraints(object):
    """
        :param geo:             Handle to MovingMetasurface3D object for optimization geometry
        :param radius_type:     Flag determining if calculating mean or max of ellipses' two radii. Valid inputs 'mean' or 'max'
        :param manual_jac:      Boolean determining whether to provide manual jacobian calculation or to use finite differences
        :param print_warning:   Boolean determining whether to print when a constraint has a value < 0
    """

    def __init__(self, geo, radius_type = 'mean', manual_jac = True, print_warning = True):
        '''The constructor for the PillarConstraints class'''

        self.geo = geo
        self.num_pillars = geo.offset_x.size
        self.print_warning = print_warning
        self.radius_type = radius_type
        self.manual_jac = True
        if not (self.radius_type == 'mean' or self.radius_type == 'max'):
            raise UserWarning("Valid radius types are 'mean' and 'max'")
    
    def get_pair_iterator(self):
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
        '''Returns jacobian on same scale as the scaled constraint'''
        offset_x, offset_y, rx, ry, phi = self.geo.get_from_params(params)
        x = offset_x + self.geo.init_x
        y = offset_y + self.geo.init_y
        N = x.size
        jac = []
        for pair in self.get_pair_iterator():
            i, j = pair[0], pair[1]
            dx, dy, drx, dry, dphi = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
            denom = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)

            dx[i] = (x[i] - x[j])/denom
            dx[j] = (x[j] - x[i])/denom

            dy[i] = (y[i] - y[j])/denom
            dy[j] = (y[j] - y[i])/denom

            if self.radius_type == 'max':
                drx[i] = -1 if rx[i] > ry[i] else 0
                drx[j] = -1 if rx[j] > ry[j] else 0
                dry[i] = -1 if ry[i] > rx[i] else 0
                dry[j] = -1 if ry[j] > rx[j] else 0
            else:
                drx[i] = -0.5
                drx[j] = -0.5
                dry[i] = -0.5
                dry[j] = -0.5

            jac.append(np.concatenate((dx, dy, drx, dry, dphi)))

        return np.stack(jac)

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