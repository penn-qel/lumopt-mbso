import nlopt
import numpy as np

from lumopt.optimizers.maximizer import Maximizer

class NLoptOptimizer(Maximizer):
    """ Wrapper for the optimizers in the NLopt package:

        https://nlopt.readthedocs.io/en/latest/

        Parameters
        --------------
        :param constraints:     Handle to constraints object
        :param method:         Chosen algorithm using nlopt.[algorithm] syntax. Default to LD_MMA
        :param scaling_factor: scalar or a vector of the same length as the optimization parameters; typically used to scale the optimization
                               parameters so that they have magnitudes in the range zero to one.
        :param scale_initial_gradient_to: enforces a rescaling of the gradient to change the optimization parameters by at least this much;
                                          the default value of zero disables automatic scaling.
        :param: penalty_fun:   penalty function to be added to the figure of merit; it must be a function that takes a vector with the
                               optimization parameters and returns a single value.
        :param: penalty_jac:   gradient of the penalty function; must be a function that takes a vector with the optimization parameters
                               and returns a vector of the same length.
        :param cons_tol:        Amount constraints are allowed to be violated. Default 0

        Shutoff Criteria (Optional)
        --------------
        :param max_iter:       maximum number of function evaluations
        :param stopval:         Parameter triggers optimizaiton stop once this value reached. Optional
        :param ftol_rel:        Relative tolerance on function value. Optional
        :param ftol_abs:        Absolute tolerance on function value. Optional
        :param xtol_rel:        Relative tolerance on optimization parameters. Optional
        :param xtol_abs:        Absolute tolerance on optimization parameters. Optional
    """

    def __init__(self, max_iter = None, constraints = None, method = nlopt.LD_MMA, scaling_factor = 1.0, scale_initial_gradient_to = 0, 
                    penalty_fun = None, penalty_jac = None, logging_path = None,
                    stopval = None, ftol_rel = None, ftol_abs = None, xtol_rel = None, xtol_abs = None, cons_tol = 0):
        super().__init__(max_iter = max_iter,
                             scaling_factor = scaling_factor,
                             scale_initial_gradient_to = scale_initial_gradient_to,
                             penalty_fun = penalty_fun,
                             penalty_jac = penalty_jac)

        self.method = method
        self.constraints = constraints
        self.stopval = stopval
        self.ftol_rel = ftol_rel
        self.ftol_abs = ftol_abs
        self.xtol_rel = xtol_rel
        self.xtol_abs = xtol_abs
        self.cons_tol = cons_tol

    def nlopt_objective(self, x, grad):
        '''Objective function in format for NLopt. Returns FOM and modifies grad in place, then runs callback'''
        #Get fom/jac
        fom = self.callable_fom(x)
        grad[:] = self.callable_jac(x)
        #Run callback
        self.callback(x)

        #Returns fom cast as scalar instead of 1-element array
        return fom.item()

    def set_tolerances(self, opt):
        '''Sets optional shutoff tolerances'''
        if self.stopval is not None:
            opt.set_stopval(self.stopval)
        if self.ftol_rel is not None:
            opt.set_ftol_rel(self.ftol_rel)
        if self.ftol_abs is not None:
            opt.set_ftol_abs(self.ftol_abs)
        if self.xtol_rel is not None:
            opt.set_xtol_rel(self.xtol_rel)
        if self.xtol_abs is not None:
            opt.set_xtol_abs(self.xtol_abs)
        if self.max_iter is not None:
            opt.set_maxeval(self.max_iter)

    def set_bounds(self, opt):
        '''Takes stored list of tuples and returns upper and lower bounds as arrays'''
        lb = np.empty(len(self.bounds))
        ub = np.empty(len(self.bounds))

        for i, bound in enumerate(self.bounds):
            lb[i] = bound[0]
            ub[i] = bound[1]

        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)

    def vector_constraints(self, result, x, grad):
        '''Defines constraint function fitting with necessary format'''
        print("Calculating constraints")
        result[:] = -1*self.constraints.scaled_constraint(x)
        grad[:] = -1*self.constraints.scaled_jacobian(x).toarray()

    def set_constraints(self, opt):
        '''Wraps constraint object into correct definition. Constraints defined to be valid when less than 0'''
        m = self.constraints.num_constraints
        tol = self.cons_tol*np.ones(m)
        opt.add_inequality_mconstraint(lambda result, x, grad: self.vector_constraints(result, x, grad), tol)

    def print_return_code(self, code):
        if code == 1:
            print("EXIT CODE 1: NLOPT SUCCESS")
            return
        if code == 2:
            print("EXIT CODE 2: NLOPT STOPVAL REACHED")
            return
        if code == 3:
            print("EXIT CODE 3: NLOPT FTOL REACHED")
            return
        if code == 4:
            print("EXIT CODE 4: NLOPT XTOL REACHED")
            return
        if code == 5:
            print("EXIT CODE 5: NLOPT MAXEVAL REACHED")
            return
        if code == 6:
            print("EXIT CODE 6: NLOPT MAXTIME REACHED")
            return
        print("WARN: NLopt ended with Unknown exit code")

    def run(self):
        '''Creates and runs optimizer'''
        #Set parameters
        opt = nlopt.opt(self.method, self.start_point.size)
        print('Running NLopt optimization using {} with {} parameters'.format(opt.get_algorithm_name(), opt.get_dimension()))
        self.set_tolerances(opt)
        opt.set_max_objective(self.nlopt_objective)
        self.set_bounds(opt)
        if self.constraints is not None:
            self.set_constraints(opt)

        #Run optimizer
        xopt = opt.optimize(self.start_point)

        #Parse results
        opt_val = opt.last_optimum_value()
        print('FINAL FOM = {}'.format(opt_val))
        self.print_return_code(opt.last_optimize_result())