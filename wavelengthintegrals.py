import numpy as np
import lumapi

#Modified from source to normalize by weights
def fom_gradient_wavelength_integral_impl(T_fwd_vs_wavelength, T_fwd_partial_derivs_vs_wl, target_T_fwd_vs_wavelength, wl, norm_p, target_T_fwd_weights):

    if wl.size > 1:
        assert T_fwd_partial_derivs_vs_wl.shape[1] == wl.size
        
        weight_norm = np.trapz(y = target_T_fwd_weights, x=wl)
        T_fwd_error = T_fwd_vs_wavelength - target_T_fwd_vs_wavelength
        #T_fwd_error = np.multiply(target_T_fwd_weights, T_fwd_error)
        T_fwd_error_integrand = target_T_fwd_weights*np.power(np.abs(T_fwd_error), norm_p) / weight_norm
        const_factor = -1.0 * np.power(np.trapz(y = T_fwd_error_integrand, x = wl), 1.0 / norm_p - 1.0)
        integral_kernel = target_T_fwd_weights*np.power(np.abs(T_fwd_error), norm_p - 1) * np.sign(T_fwd_error) / weight_norm
        
        ## Implement the trapezoidal integration as a matrix-vector-product for performance reasons
        d = np.diff(wl)
        quad_weight = np.append(np.append(d[0], d[0:-1]+d[1:]),d[-1])/2 #< There is probably a more elegant way to do this
        v = const_factor * integral_kernel * quad_weight
        T_fwd_partial_derivs = T_fwd_partial_derivs_vs_wl.dot(v)

    else:
        T_fwd_partial_derivs = -1.0 * np.sign(T_fwd_vs_wavelength - target_T_fwd_vs_wavelength) * T_fwd_partial_derivs_vs_wl.flatten()

    return T_fwd_partial_derivs.flatten().real

#Modified from source code to normalize by weights rather than by wavelength
def fom_wavelength_integral(T_fwd_vs_wavelength, wavelengths, target_T_fwd, norm_p, target_T_fwd_weights):
    target_T_fwd_vs_wavelength = target_T_fwd(wavelengths).flatten()
    target_T_fwd_weights_vs_wavelength = target_T_fwd_weights(wavelengths).flatten()
    if len(wavelengths) > 1:
        weight_norm = np.trapz(y = target_T_fwd_weights_vs_wavelength, x = wavelengths)
        T_fwd_integrand = np.multiply(target_T_fwd_weights_vs_wavelength, np.power(np.abs(target_T_fwd_vs_wavelength), norm_p)) / weight_norm
        #T_fwd_integrand = np.power(np.abs(target_T_fwd_vs_wavelength), norm_p) / wavelength_range
        const_term = np.power(np.trapz(y = T_fwd_integrand, x = wavelengths), 1.0 / norm_p)
        T_fwd_error = np.abs(T_fwd_vs_wavelength.flatten() - target_T_fwd_vs_wavelength)
        T_fwd_error_integrand = np.multiply(target_T_fwd_weights_vs_wavelength, np.power(T_fwd_error, norm_p)) / weight_norm
        #T_fwd_error_integrand = np.power(T_fwd_error, norm_p) / wavelength_range
        error_term = np.power(np.trapz(y = T_fwd_error_integrand, x = wavelengths), 1.0 / norm_p)
        fom = const_term - error_term
    else:
        fom = np.abs(target_T_fwd_vs_wavelength) - np.abs(T_fwd_vs_wavelength.flatten() - target_T_fwd_vs_wavelength)
    return fom.real


def fom_gradient_wavelength_integral_on_cad_impl(sim, grad_var_name, T_fwd_vs_wavelength, target_T_fwd_vs_wavelength, wl, norm_p, target_T_fwd_weights):
    weight_norm = np.trapz(y = target_T_fwd_weights, x=wl)
    T_fwd_error = T_fwd_vs_wavelength - target_T_fwd_vs_wavelength
    #T_fwd_error_integrand = target_T_fwd_weights*np.power(np.abs(T_fwd_error), norm_p) / weight_norm**************
    T_fwd_unweighted_integrand = np.power(np.abs(T_fwd_error), norm_p)

    if wl.size > 1:
        weight_norm = np.trapz(y = target_T_fwd_weights, x=wl)
        T_fwd_error_integrand = target_T_fwd_weights*np.power(np.abs(T_fwd_error), norm_p) / weight_norm
        const_factor = -1.0 * np.power(np.trapz(y = T_fwd_error_integrand, x = wl), 1.0 / norm_p - 1.0)
        integral_kernel = target_T_fwd_weights*np.power(np.abs(T_fwd_error), norm_p - 1) * np.sign(T_fwd_error) / weight_norm
        
        d = np.diff(wl)
        quad_weight = np.append(np.append(d[0], d[0:-1]+d[1:]),d[-1])/2 #< There is probably a more elegant way to do this
        v = const_factor * integral_kernel * quad_weight

        lumapi.putMatrix(sim.fdtd.handle, "wl_scaled_integral_kernel", v)
        sim.fdtd.eval(('dF_dp_s=size({0});'
                       'dF_dp2 = reshape(permute({0},[3,2,1]),[dF_dp_s(3),dF_dp_s(2)*dF_dp_s(1)]);'
                       'T_fwd_partial_derivs=real(mult(transpose(wl_scaled_integral_kernel),dF_dp2));').format(grad_var_name) )
        T_fwd_partial_derivs_on_cad = sim.fdtd.getv("T_fwd_partial_derivs")

    else:
        sim.fdtd.eval(('T_fwd_partial_derivs=real({0});').format(grad_var_name) )
        T_fwd_partial_derivs_on_cad = sim.fdtd.getv("T_fwd_partial_derivs")
        T_fwd_partial_derivs_on_cad*= -1.0 * np.sign(T_fwd_error)

    return T_fwd_partial_derivs_on_cad.flatten()