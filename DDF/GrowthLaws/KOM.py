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
import growth_laws as gl
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

'''Kinematics'''
I = ufl.Identity(len(u))
F = ufl.variable(I + ufl.grad(u))

F_g_tot_function, F_g_tot_expression, E_e_function, E_e_expression, F_e = gl.KOM(mesh, F)

C_e = F_e.T*F_e
J_e = ufl.det(F_e)

'''Constants'''
mu = dolfinx.default_scalar_type(1)
kappa = dolfinx.default_scalar_type(1e4)

'''Create compressible strain energy function'''
psi_inc  = psi.neohookean(mu/2, C_e)
psi_comp = psi.comp2(kappa, J_e) 
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
us = []; F_g_f_tot = []; F_g_c_tot = []; F_e11_list = []; F_e22_list = []; F_e33_list = []; J_e = []; J_g = []; J_g_tot = []; J_tot = []; max_strn = []; k_growth_list1 = []; k_growth_list2 = []; E_e00_list = []; E_e11_list = []; F_gf = []; F_gc = []; 

'''Solve The Problem'''
N = 2000   # Number of time steps
for i in range(0, N+1):

    if i % 100 == 0:
        print(f"Time step {i}/{N}")
        # breakpoint()

    # Tabulate values for postprocessing
    if i % 10 == 0:
        u_new = u.copy()
        us.append(u_new)    
        F_g_f_tot.append(ddf.eval_expression(F_g_tot_function[0,0], mesh)[0,0])
        F_g_c_tot.append(ddf.eval_expression(F_g_tot_function[1,1], mesh)[0,0])
    
    solver.solve(u)     # Solve the problem

    F_g_tot_function.interpolate(F_g_tot_expression)        # Update total growth tensor
    E_e_function.interpolate(E_e_expression)                # Update elastic deformation tensor

'''Write to file to plot in Desmos and Paraview'''
lists_to_write = {
    "F_{totgff}": F_g_f_tot,
    "F_{totgcc}": F_g_c_tot
}

pp.write_lists_to_file("simulation_results.txt", lists_to_write)
pp.write_vector_to_paraview("../ParaViewData/simple_growth_meeting.xdmf", mesh, us)