import numpy as np 
from typing import NamedTuple, Callable
import ufl
import dolfinx as df
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from pathlib import Path
from enum import Enum
from dolfinx.io import XDMFFile
from petsc4py import PETSc
import basix

sys.path.insert(1, '/MyProject/MyCode/DDF/helper')

import geometry as geo
import ddf as ddf 
import postprocessing as pp
import hyperelastic_models as psi
import simple_model

class Neumann(NamedTuple):
    marker: int
    locator: Callable[[np.ndarray], np.ndarray]
    value: df.fem.Constant

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

dirichlet_bc = []
x_left  = (lambda x : np.isclose(x[0], x_min), 0, df.default_scalar_type(0))
y_left  = (lambda x : np.isclose(x[1], y_min), 1, df.default_scalar_type(0))
z_left  = (lambda x : np.isclose(x[2], z_min), 2, df.default_scalar_type(0))
z_right = (lambda x : np.isclose(x[2], z_max), 2, df.default_scalar_type(0))

# u_ex = lambda x: x
neumann_bc = df.fem.Constant(mesh, -2.0)
# neumann_bc += 1
y_right_neumann = Neumann(1, lambda x : np.isclose(x[1], y_max), neumann_bc)

neumann_bc_values = [y_right_neumann]


bc_values = [x_left, y_left, z_left, z_right]

'''Create Stress Tensor'''
infinitesimal_strain = 1/2*(ufl.grad(u).T * ufl.grad(u))
gamma = ufl.sqrt(2)

'''From the paper'''
V = df.fem.FunctionSpace(mesh, ("Discontinuous Lagrange", 0))
eff = df.fem.Function(V)
F_g = simple_model.F_g(gamma)

invF_g = ufl.inv(F_g)

I = ufl.Identity(len(u))                        # Identity tensor
F = ufl.variable(I + ufl.grad(u))               # Deformation tensor
F_e = ufl.variable(F*invF_g)#*pow(J, -1 / 3)#                  # Elasticity tensor
F_e_function, F_e_expression = ddf.to_tensor_map(F_e, mesh)
F_function, F_expression = ddf.to_tensor_map(F_e*F_g, mesh)

J = ufl.variable(ufl.det(F_e))                                # Determinant

E = 1/2*(F_e.T*F_e - I)                         # Difference tensor
C = F_e.T * F_e

'''Constants'''

kappa = 4e1
W_inc  = (ufl.tr(C) - 3)
W_comp = kappa*(pow((J - 1), 2))
W =  W_inc + W_comp
P = ufl.diff(W, F_e)
T_function, T_expression = ddf.to_tensor_map(P, mesh)

weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions And Set Up Solver'''
bcs, boundary_points = ddf.dirichlet_injection(mesh, bc_values, function_space)

breakpoint()
P_with_bcs = ddf.neumann_injection(mesh, neumann_bc_values, weak_form, v)
breakpoint()

mesh.topology.create_connectivity(
    mesh.topology.dim-1, mesh.topology.dim
    )

for name, tags in boundary_points.items():
    facet_file = Path(f"ParaViewData/facet_tags_{name}.xdmf")
    facet_file.unlink(missing_ok=True)
    facet_file.with_suffix(".h5").unlink(missing_ok=True)
    with XDMFFile(mesh.comm, facet_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(tags, mesh.geometry)

us = []
u_new = u.copy()
us.append(u_new)
T_s = []
T_function.interpolate(T_expression)
T_new = T_function.copy()
T_s.append(T_new)

print("Creating problem")
problem = NonlinearProblem(P_with_bcs, u, bcs)
print("Creating solver")
solver = NewtonSolver(mesh.comm, problem)
print("Solving")
solver.solve(u)
u_new = u.copy()
us.append(u_new)
T_function.interpolate(T_expression)
T_new = T_function.copy()
T_s.append(T_new)

# solver.max_it = 500
print("Done solving")

pp.write_vector_to_paraview("ParaViewData/simple_growth.xdmf", mesh, us)
pp.write_tensor_to_paraview("ParaViewData/T.xdmf", mesh, T_s)