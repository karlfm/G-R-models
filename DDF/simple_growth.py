import numpy as np 
import ufl
import dolfinx
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from pathlib import Path
from enum import Enum
from dolfinx.io import XDMFFile
from petsc4py import PETSc
import basix
import cardiac_geometries

sys.path.insert(1, '/home/shared/helper')

import geometry as geo
import ddf as ddf 
import postprocessing as pp
import hyperelastic_models as psi
import growth_laws

#region
'''Create Geometry'''
x_min, x_max, Nx = 0, 1, 8
y_min, y_max, Ny = 0, 1, 8
z_min, z_max, Nz = 0, 1, 8
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

'''Get Functions'''
function_space, u, v = ddf.get_functions(mesh)

x_left_x  = (lambda x : np.isclose(x[0], x_min), 0, dolfinx.default_scalar_type(0))
x_right_x = (lambda x : np.isclose(x[0], x_max), 0, dolfinx.default_scalar_type(0.2))
x_left_y  = (lambda x : np.isclose(x[0], x_min), 1, dolfinx.default_scalar_type(0))
x_right_y = (lambda x : np.isclose(x[0], x_max), 1, dolfinx.default_scalar_type(0))
x_left_z  = (lambda x : np.isclose(x[0], x_min), 2, dolfinx.default_scalar_type(0))
x_right_z = (lambda x : np.isclose(x[0], x_max), 2, dolfinx.default_scalar_type(0))
y_left    = (lambda x : np.isclose(x[1], y_min), 1, dolfinx.default_scalar_type(0))
z_left    = (lambda x : np.isclose(x[2], z_min), 2, dolfinx.default_scalar_type(0))
z_right   = (lambda x : np.isclose(x[2], z_max), 2, dolfinx.default_scalar_type(0))

bc_values = [x_left_x, x_left_y, x_left_z, x_right_x, x_right_y, x_right_z]#, y_left, z_left, z_right]
#bc_values  = [x_right]

'''Create Stress Tensor'''
strain = 1/2*(ufl.grad(u).T*ufl.grad(u) + ufl.grad(u).T + ufl.grad(u))


X = ufl.SpatialCoordinate(mesh)     # get Identity without it being a constant
F_0 = ufl.grad(X)                   # --//--    
F_0_func, F0_form = ddf.to_tensor_map(F_0, mesh)
V = dolfinx.fem.functionspace(mesh, basix.ufl.element(family="DG", cell=str(mesh.ufl_cell()), degree=0, shape=(3,3)))
# F_0_func = dolfinx.fem.Function(V)       # this gets updated in the loop
# F0_form = dolfinx.fem.Expression(F_0, V.element.interpolation_points())  # 
F_0_func.interpolate(F0_form)

strain_function, strain_expression = ddf.to_tensor_map(strain, mesh)

'''From the paper'''
I = ufl.Identity(len(u))
F = ufl.variable(I + ufl.grad(u))
F_g = growth_laws.F_g1(strain_function, F_0_func) * F_0_func
# I don't understand why the next line of code is needed
# I think it is because we need to initiate the first previous total growth step to create the growth accumulation tensor
strain_function.interpolate(strain_expression)
F_g_prev = growth_laws.F_g1(strain_function, F_0_func)
F_e = ufl.variable(F*ufl.inv(F_g))

J = ufl.variable(ufl.det(F_e))

F_e_bar = F_e*pow(J, -1/3)
C_bar = F_e_bar.T*F_e_bar

'''Constants'''
El = dolfinx.default_scalar_type(1.0e4)
nu = dolfinx.default_scalar_type(0.3)
mu = dolfinx.fem.Constant(mesh, El / (2 * (1 + nu)))
lmbda = dolfinx.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))
kappa = 1e6

'''Create compressible strain energy function'''
psi_inc  = psi.neohookean(mu/2, C_bar)
psi_comp = psi.comp2(kappa, J) 
psi_=  psi_inc + psi_comp
P = ufl.diff(psi_, F_e)*F_e.T/J

'''Create weak form'''
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions'''
bcs, _ = ddf.dirichlet_injection(mesh, bc_values, function_space)

'''Assemble FEniCS solver'''
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(mesh.comm, problem)

us = []

# solver.solve(u)
# u_new = u.copy()
# us.append(u_new)



'''Solve The Problem'''
for i in np.arange(16):

    print(i)

    # print(ddf.eval_expression(pow((1/2*(ufl.sqrt(2*strain_function[1,1] + 1) - 1/2 - 1) + 1), 1/3), mesh))

    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)
    #breakpoint()
    ddf.eval_expression(strain, mesh)

    # F_0_func creates an expression for F_g and then evaluates it (by using .interpolate()).
    # Since F_0_func is part of the input for F_g; F_g updates everytime.
    F_0_func.interpolate(dolfinx.fem.Expression(F_g_prev, V.element.interpolation_points()))
    
    strain_function.interpolate(strain_expression)

pp.write_vector_to_paraview("ParaViewData/simple_growth.xdmf", mesh, us)