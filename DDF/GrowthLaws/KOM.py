import numpy as np 
import ufl
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from dolfinx.io import XDMFFile
from petsc4py import PETSc
import basix

sys.path.insert(1, '/home/shared/helper')

import geometry as geo
import ddf as ddf 
import postprocessing as pp
import hyperelastic_models as psi
#region

'''Create Geometry'''
x_min, x_max, Nx = 0, 1, 4
y_min, y_max, Ny = 0, 1, 4
z_min, z_max, Nz = 0, 1, 4
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

'''Get Functions'''
vector_space, u, v = ddf.get_vector_functions(mesh)

'''Boundary Conditions'''
#Dirichlet
x_left_x  = (lambda x : np.isclose(x[0], x_min), 0, dolfinx.default_scalar_type(0))
x_right_x = (lambda x : np.isclose(x[0], x_max), 0, dolfinx.default_scalar_type(0.1))
x_left_y  = (lambda x : np.isclose(x[0], x_min), 1, dolfinx.default_scalar_type(0))
x_right_y = (lambda x : np.isclose(x[0], x_max), 1, dolfinx.default_scalar_type(0))
x_left_z  = (lambda x : np.isclose(x[0], x_min), 2, dolfinx.default_scalar_type(0))
x_right_z = (lambda x : np.isclose(x[0], x_max), 2, dolfinx.default_scalar_type(0))
y_left_y  = (lambda x : np.isclose(x[1], y_min), 1, dolfinx.default_scalar_type(0))
z_left_z  = (lambda x : np.isclose(x[2], z_min), 2, dolfinx.default_scalar_type(0))
z_right   = (lambda x : np.isclose(x[2], z_max), 2, dolfinx.default_scalar_type(0))

bc_values  = [x_left_x, y_left_y, z_left_z, x_right_x]

neumann_x_right = (1, lambda x : np.isclose(x[0], x_max), dolfinx.default_scalar_type(0.1))
natural_bcs = []

'''Initiate first growth tensor'''
tensor_space = dolfinx.fem.functionspace(mesh, basix.ufl.element(family="DG", cell=str(mesh.ufl_cell()), degree=0, shape=(3,3)))
X = ufl.SpatialCoordinate(mesh)       # get Identity without it being a constant
Identity = ufl.variable(ufl.grad(X)) 
Identity_expression = dolfinx.fem.Expression(Identity, tensor_space.element.interpolation_points())

F_g_tot_function = dolfinx.fem.Function(tensor_space); F_g_tot_function.interpolate(Identity_expression)
E_e_function = dolfinx.fem.Function(tensor_space); E_e_function.interpolate(Identity_expression)

'''Kinematics'''
I = ufl.Identity(len(u))
F = ufl.variable(I + ufl.grad(u))

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

# Expressions used to update total growth tensor and elastic deformation tensor
F_g_tot_expression = dolfinx.fem.Expression(F_g*F_g_tot_function, tensor_space.element.interpolation_points())
E_e_expression = dolfinx.fem.Expression(0.5*(F_e.T*F_e - ufl.Identity(3)), tensor_space.element.interpolation_points())

# Determinant and right Cauchy-Green tensor
J = ufl.variable(ufl.det(F_e))
C_e = F_e.T*F_e

'''Constants'''
mu = dolfinx.default_scalar_type(15)
kappa = dolfinx.default_scalar_type(1e4)

'''Create compressible strain energy function'''
psi_inc  = psi.neohookean(mu/2, C_e)
psi_comp = psi.comp2(kappa, J) 
psi_=  psi_inc + psi_comp
P = ufl.diff(psi_, F_e)

'''Create weak form'''
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions'''
bcs, _ = ddf.dirichlet_injection(mesh, bc_values, vector_space)
if natural_bcs:     #checks if natrual_bcs list is empty
    natural_bcs_applied = ddf.natural_injection(mesh, natural_bcs, F_e, v, u)
    
    for bc in natural_bcs_applied:
        weak_form -= bc

'''Assemble FEniCS solver'''
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(mesh.comm, problem)

# Preallocate lists for postprocessing
us = []; F_g_f_tot = []; F_g_c_tot = []; F_e11_list = []; F_e22_list = []; F_e33_list = []; J_e = []; J_g = []; J_g_tot = []; J_tot = []

'''Solve The Problem'''
N = 10000   # Number of time steps
for i in range(0, N+1):

    # Tabulate values for postprocessing
    if i % 50 == 0:
        print(f"Time step {i}/{N}")
        u_new = u.copy()
        us.append(u_new)    
        F_g_f_tot.append(ddf.eval_expression(F_g_tot_function[0,0], mesh)[0,0])
        F_g_c_tot.append(ddf.eval_expression(F_g_tot_function[1,1], mesh)[0,0])
        F_e11_list.append(ddf.eval_expression(F_e[0,0], mesh)[0,0])
        F_e22_list.append(ddf.eval_expression(F_e[1,1], mesh)[0,0])
        F_e33_list.append(ddf.eval_expression(F_e[2,2], mesh)[0,0])

        J_e.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form((ufl.det(F_e))*ufl.dx(metadata={"quadrature_degree": 8}))))
        J_g.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form((ufl.det(F_g))*ufl.dx(metadata={"quadrature_degree": 8}))))
        J_tot.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form((ufl.det(F))*ufl.dx(metadata={"quadrature_degree": 8}))))
        J_g_tot.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form((ufl.det(F_g_tot_function))*ufl.dx(metadata={"quadrature_degree": 8}))))
    
    solver.solve(u)     # Solve the problem

    F_g_tot_function.interpolate(F_g_tot_expression)        # Update total growth tensor
    E_e_function.interpolate(E_e_expression)                # Update elastic deformation tensor

'''Write to file to plot in Desmos and Paraview'''
lists_to_write = {
    "J_e": J_e,
    "J_g": J_g,
    "J": J_tot,
    "J_{gtot}": J_g_tot,
    "F_{gff}": F_g_f_tot,
    "F_{gcc}": F_g_c_tot,
    "F_{e11}": F_e11_list,
    "F_{e22}": F_e22_list,
    "F_{e33}": F_e33_list
}

pp.write_lists_to_file("simulation_results.txt", lists_to_write)
pp.write_vector_to_paraview("../ParaViewData/simple_growth_meeting.xdmf", mesh, us)