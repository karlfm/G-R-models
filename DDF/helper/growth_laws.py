import dolfinx
import ufl
import basix

def KOM(mesh, F):

    tensor_space = dolfinx.fem.functionspace(mesh, basix.ufl.element(family="DG", cell=str(mesh.ufl_cell()), degree=0, shape=(3,3)))
    X = ufl.SpatialCoordinate(mesh)       # get Identity without it being a constant
    Identity = ufl.variable(ufl.grad(X)) 
    Identity_expression = dolfinx.fem.Expression(Identity, tensor_space.element.interpolation_points())

    F_g_tot_function = dolfinx.fem.Function(tensor_space); F_g_tot_function.interpolate(Identity_expression)
    E_e_function = dolfinx.fem.Function(tensor_space); E_e_function.interpolate(Identity_expression)

    '''Constants from the paper'''
    f_ff_max    = 0.3
    f_f         = 150   
    s_l50       = 0.06
    F_ff50      = 1.35
    f_l_slope   = 40
    f_cc_max    = 0.1
    c_f         = 75
    s_t50       = 0.07
    F_cc50      = 1.28
    c_th_slope  = 60

    '''Growth Laws'''
    def k_growth(F_g_cum: dolfinx.fem.Function, slope: int, F_50: dolfinx.fem.Function) -> dolfinx.fem.Function:
        return 1 / (1 + ufl.exp(slope * (F_g_cum - F_50)))

    def alg_max_princ_strain(E: dolfinx.fem.Function) -> dolfinx.fem.Function:
        return (E[1,1] + E[2,2])/2 + ufl.sqrt(((E[1,1] - E[2,2])/2)**2 + (E[1,2]*E[2,1]))

    dt = 0.1
    # Growth in the fiber direction
    F_gff = ufl.conditional(ufl.ge(E_e_function[0,0], 0), 
                            k_growth(F_g_tot_function[0,0], f_l_slope, F_ff50)*f_ff_max*dt/(1 + ufl.exp(-f_f*(E_e_function[0,0] - s_l50))) + 1, 
                            -f_ff_max*dt/(1 + ufl.exp(f_f*(E_e_function[0,0] + s_l50))) + 1)

    # Growth in the cross-fiber direction
    F_gcc = ufl.conditional(ufl.ge(alg_max_princ_strain(E_e_function), 0), 
                            ufl.sqrt(k_growth(F_g_tot_function[1,1], c_th_slope, F_cc50)*f_cc_max*dt/(1 + ufl.exp(-c_f*(alg_max_princ_strain(E_e_function) - s_t50))) + 1), 
                            ufl.sqrt(-f_cc_max*dt/(1 + ufl.exp(c_f*(alg_max_princ_strain(E_e_function) + s_t50))) + 1))

    # Incremental growth tensor
    F_g = ufl.as_tensor((
        (F_gff, 0, 0),
        (0, F_gcc, 0),
        (0, 0, F_gcc)))

    # Elastic deformation tensor
    F_e = ufl.variable(F*ufl.inv(F_g_tot_function))

    F_g_tot_expression = dolfinx.fem.Expression(F_g*F_g_tot_function, tensor_space.element.interpolation_points())
    E_e_expression = dolfinx.fem.Expression(0.5*(F_e.T*F_e - ufl.Identity(3)), tensor_space.element.interpolation_points())

    return F_g_tot_function, F_g_tot_expression, E_e_function, E_e_expression, F_e