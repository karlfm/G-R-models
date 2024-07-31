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
x_min, x_max, Nx = 0, 1, 4
y_min, y_max, Ny = 0, 1, 4
z_min, z_max, Nz = 0, 1, 4
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

'''Get Functions'''
function_space, u, v = ddf.get_functions(mesh)

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

# bc_values = [x_left_x, x_left_y, x_left_z, x_right_x, x_right_y, x_right_z]#, y_left_y, z_left_z]#, x_right_x, x_right_y, x_right_z]#, y_left, z_left, z_right]
# bc_values = [x_left_x, x_left_y, x_left_z]
bc_values  = [x_left_x, x_right_x, y_left_y, z_left_z]

'''Neumann'''
neumann_x_right_x  = (1, lambda x : np.isclose(x[0], x_max), dolfinx.fem.Constant(mesh, 5000.0))
neumann_bc_values = []

'''Robin'''
robin_x_right_x  = (2, lambda x : np.isclose(x[0], x_max), (dolfinx.fem.Constant(mesh, 1.0), dolfinx.fem.Constant(mesh, -5000.0)))
robin_bc_values = []

natural_bcs = robin_bc_values + neumann_bc_values

'''Initiate first growth tensor'''
X = ufl.SpatialCoordinate(mesh)       # get Identity without it being a constant
F_g0 = ufl.variable(ufl.grad(X))      # --//--    

# Create function space to evaluate functions in -> create an expression -> create a function -> interpolate the expression
function_space2 = dolfinx.fem.functionspace(mesh, basix.ufl.element(family="CG", cell=str(mesh.ufl_cell()), degree=2, shape=(3,3)))
F_g_expression = dolfinx.fem.Expression(F_g0, function_space2.element.interpolation_points())
F_g_function = dolfinx.fem.Function(function_space2)

F_g_function.interpolate(F_g_expression)


'''Kinematics'''
I = ufl.Identity(len(u))
F = ufl.variable(I + ufl.grad(u))
F_e = ufl.variable(F*ufl.inv(F_g_function))

E = 1/2*(F_e.T*F_e - ufl.Identity(3))
stretch = (F.T*F)
set_point = -0.0

'''Constants'''
El = dolfinx.default_scalar_type(1.0e4)
nu = dolfinx.default_scalar_type(0.3)
mu = dolfinx.fem.Constant(mesh, El / (2 * (1 + nu)))
lmbda = dolfinx.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))
kappa = 1e6

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


E_ff_expression = dolfinx.fem.Expression(E[0,0], function_space2.element.interpolation_points())
E_cc_expression = dolfinx.fem.Expression(E[1,1], function_space2.element.interpolation_points())
E_ff = E[0,0]
E_cc = E[1,1]
s_l = E_ff - 1
s_t = E_cc - 1

k_ff = 1/(1 + ufl.exp(f_lengthslope*(F_g_function[0,0] - F_ff50)))
k_cc = 1/(1 + ufl.exp(c_thicknessslope*(F_g_function[1,1] - F_cc50)))

F_gff = ufl.conditional(ufl.ge(s_l, 0), k_ff*f_ffmax*dt/(1 + ufl.exp(-f_f*(s_l - s_l50))) + 1, -f_ffmax*dt/(1 + ufl.exp(f_f*(s_l + s_l50))) + 1)
F_gcc = ufl.conditional(ufl.ge(s_t, 0), ufl.sqrt(k_cc*f_ffmax*dt/(1 + ufl.exp(-c_f*(s_t - s_t50))) + 1), ufl.sqrt(-f_ffmax*dt/(1 + ufl.exp(c_f*(s_t + s_t50))) + 1))

F_g = ufl.as_tensor((
    (F_gff, 0, 0),
    (0, F_gcc, 0),
    (0, 0, F_gcc)))

F_g_succ_expression = dolfinx.fem.Expression(F_g*F_g_function, function_space2.element.interpolation_points())

#region
J = ufl.variable(ufl.det(F_e))

F_e_bar = F_e*pow(J, -1/3)
C_bar = F_e_bar.T*F_e_bar

'''Create compressible strain energy function'''
psi_inc  = psi.neohookean(mu/2, C_bar)
psi_comp = psi.comp2(kappa, J) 
psi_=  psi_inc + psi_comp
P = ufl.diff(psi_, F_e)*F_e.T/J

'''Create weak form'''
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions'''
if bc_values:
    bcs, _ = ddf.dirichlet_injection(mesh, bc_values, function_space)
else:
    print("You have not implemented essential (Dirichlet) boundary conditions")


if natural_bcs:     #is true if natural_bcs is a non-empty list
    natural_bcs_applied = ddf.natural_injection(mesh, natural_bcs, F, v, u)
    for bc in natural_bcs_applied:
        weak_form -= bc
else:
    print("You have not implemented natural (Neumann/Robin) boundary conditions.")

'''Assemble FEniCS solver'''
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(mesh.comm, problem)

us = []
'''Solve once to get set point values etc.'''
solver.solve(u)
u_new = u.copy()
us.append(u_new)

#endregion

'''Solve The Problem'''
n = 7
for i in np.arange(n):
    print("\n", i, "/", n-1, "\n")
    # print(ddf.eval_expression(stretch, mesh))
    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)

    F_g_function.interpolate(F_g_succ_expression)


pp.write_vector_to_paraview("ParaViewData/simple_growth.xdmf", mesh, us)