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
from typing import Optional

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
vector_space, u, v = ddf.get_functions(mesh)

'''Boundary Conditions'''
'''Dirichlet'''
x_left_x  = (lambda x : np.isclose(x[0], x_min), 0, dolfinx.default_scalar_type(0))
x_right_x = (lambda x : np.isclose(x[0], x_max), 0, dolfinx.default_scalar_type(0.1))
x_left_y  = (lambda x : np.isclose(x[0], x_min), 1, dolfinx.default_scalar_type(0))
x_right_y = (lambda x : np.isclose(x[0], x_max), 1, dolfinx.default_scalar_type(0))
x_left_z  = (lambda x : np.isclose(x[0], x_min), 2, dolfinx.default_scalar_type(0))
x_right_z = (lambda x : np.isclose(x[0], x_max), 2, dolfinx.default_scalar_type(0))
y_left_y  = (lambda x : np.isclose(x[1], y_min), 1, dolfinx.default_scalar_type(0))
z_left_z  = (lambda x : np.isclose(x[2], z_min), 2, dolfinx.default_scalar_type(0))
z_right   = (lambda x : np.isclose(x[2], z_max), 2, dolfinx.default_scalar_type(0))

bc_values  = [x_left_x, x_right_x, y_left_y, z_left_z]

'''Initiate first growth tensor'''
tensor_space = dolfinx.fem.functionspace(mesh, basix.ufl.element(family="CG", cell=str(mesh.ufl_cell()), degree=2, shape=(3,3)))
X = ufl.SpatialCoordinate(mesh)       # get Identity without it being a constant
F_g0 = ufl.variable(ufl.grad(X))      # --//--    
F_g_expression = dolfinx.fem.Expression(F_g0, tensor_space.element.interpolation_points())
F_g_function = dolfinx.fem.Function(tensor_space)
F_g_function.interpolate(F_g_expression)
F_g_tot_function = dolfinx.fem.Function(tensor_space)
F_g_tot_function.interpolate(F_g_expression)

'''Kinematics'''
I = ufl.Identity(len(u))
F = ufl.variable(I + ufl.grad(u))

'''Constants from the paper'''
#region

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
#endregion

def k_growth(F_g_cum: dolfinx.fem.Function, slope: int, F_50: dolfinx.fem.Function) -> dolfinx.fem.Function:
    return 1 / (1 + ufl.exp(slope * (F_g_cum - F_50)))

def alg_max_princ_strain(E: dolfinx.fem.Function) -> dolfinx.fem.Function:
    return (E[1,1] + E[2,2])/2 + ufl.sqrt(((E[1,1] - E[2,2])/2)**2 + (E[1,2]*E[2,1]))
    # return E[1,1]
s_function = dolfinx.fem.Function(tensor_space)

dt = 0.1

t=dolfinx.fem.Constant(mesh, 0.0)

F_gff = ufl.conditional(ufl.gt(t, 0),
                    ufl.conditional(ufl.ge(s_function[0,0], 0), 
                    k_growth(F_g_tot_function[0,0], f_l_slope, F_ff50)*f_ff_max*dt/(1 + ufl.exp(-f_f*(s_function[0,0] - s_l50))) + 1, 
                    -f_ff_max*dt/(1 + ufl.exp(f_f*(s_function[0,0] + s_l50))) + 1),
                    1.0)


F_gcc = ufl.conditional(ufl.gt(t, 0),
                    ufl.conditional(ufl.ge(alg_max_princ_strain(s_function), 0), 
                    ufl.sqrt(k_growth(F_g_tot_function[1,1], c_th_slope, F_cc50)*f_cc_max*dt/(1 + ufl.exp(-c_f*(alg_max_princ_strain(s_function) - s_t50))) + 1), 
                    ufl.sqrt(-f_cc_max*dt/(1 + ufl.exp(c_f*(alg_max_princ_strain(s_function) + s_t50))) + 1)),
                    1.0)


F_g = ufl.as_tensor((
    (F_gff, 0, 0),
    (0, F_gcc, 0),
    (0, 0, F_gcc)))

F_e = ufl.variable(F*ufl.inv(F_g_tot_function))

F_g_tot = dolfinx.fem.Expression(F_g*F_g_tot_function, tensor_space.element.interpolation_points())

s_expression = dolfinx.fem.Expression(0.5*(F_e.T*F_e - ufl.Identity(3)), tensor_space.element.interpolation_points())

#region
J = ufl.variable(ufl.det(F_e))

F_e_bar = F_e*pow(J, -1/3)
C_bar = F_e_bar.T*F_e_bar

'''Constants'''
El = dolfinx.default_scalar_type(1.0e4)
nu = dolfinx.default_scalar_type(0.3)
mu = dolfinx.fem.Constant(mesh, El / (2 * (1 + nu)))
lmbda = dolfinx.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))
kappa = 1e5

'''Create compressible strain energy function'''
psi_inc  = psi.neohookean(mu/2, C_bar)
psi_comp = psi.comp2(kappa, J) 
psi_=  psi_inc + psi_comp
P = ufl.diff(psi_, F)

'''Create weak form'''
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions'''
bcs, _ = ddf.dirichlet_injection(mesh, bc_values, vector_space)

'''Assemble FEniCS solver'''
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(mesh.comm, problem)

us = []
'''Solve once to get set point values etc.'''

#endregion

solver.solve(u)
u_new = u.copy()
us.append(u_new)

F_g_f_tot = []; F_g_c_tot = []
'''Solve The Problem'''
N = 1000
for i in range(N):

    t.value = i
 
    F_g_tot_function.interpolate(F_g_tot)
    s_function.interpolate(s_expression)

    if i % 100 == 0:
        print("Step ", i)
        print("s = ", ddf.eval_expression(s_function, mesh))
        print("F_g = ", ddf.eval_expression(F_g, mesh))
        print("F_g_tot = ", ddf.eval_expression(F_g_tot_function, mesh))
        print("F_e = ", ddf.eval_expression(F_e, mesh))

    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)

    F_g_f_tot.append(ddf.eval_expression(F_g_tot_function[0,0], mesh)[0,0])
    F_g_c_tot.append(ddf.eval_expression(F_g_tot_function[1,1], mesh)[0,0])

breakpoint()
pp.write_vector_to_paraview("ParaViewData/simple_growth.xdmf", mesh, us)