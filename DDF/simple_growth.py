import numpy as np 
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

#region
'''Create Geometry'''
x_min, x_max, Nx = 0, 1, 6
y_min, y_max, Ny = 0, 1, 6
z_min, z_max, Nz = 0, 1, 6
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

'''Get Functions'''
function_space, u, v = ddf.get_functions(mesh)
# state_space, state, u, p, v, q = solv.get_mixed_functions(mesh)
# function_space, _, _ = solv.get_functions(mesh)
grad_u_func, grad_u_expression = ddf.to_tensor_map(ufl.grad(u), mesh)

dirichlet_bc = []
x_left  = (lambda x : np.isclose(x[0], x_min), 0, df.default_scalar_type(0))
x_right = (lambda x : np.isclose(x[0], x_max), 0, df.default_scalar_type(1))
y_left  = (lambda x : np.isclose(x[1], y_min), 1, df.default_scalar_type(0))
z_left  = (lambda x : np.isclose(x[2], z_min), 2, df.default_scalar_type(0))

bc_values = [x_left, x_right, y_left, z_left]
#bc_values  = [x_right]
'''Create Stress Tensor'''
infinitesimal_strain = 1/2*(ufl.grad(u).T * ufl.grad(u))
strain_function, strain_expression = ddf.to_tensor_map(infinitesimal_strain, mesh)
strain_function_00, strain_expression_00 = ddf.to_scalar_map(infinitesimal_strain[0, 0], mesh)


'''From the paper'''
V = df.fem.FunctionSpace(mesh, ("Discontinuous Lagrange", 0))
eff = df.fem.Function(V)
F_g = simple_model.F_g(strain_function)
F_g_function, F_g_expression = ddf.to_tensor_map(F_g, mesh)
F_g_expression_old = F_g_expression

invF_g = ufl.inv(F_g)

I = ufl.Identity(len(u))                        # Identity tensor
F = ufl.variable(I + ufl.grad(u))               # Deformation tensor
F_e = ufl.variable(F*invF_g)#*pow(J, -1 / 3)#                  # Elasticity tensor
F_e_function, F_e_expression = ddf.to_tensor_map(F_e, mesh)
F_function, F_expression = ddf.to_tensor_map(F_e*F_g, mesh)

J = ufl.variable(ufl.det(F_e))                                # Determinant
F_e_bar = F_e*pow(J, -1/3)

E = 1/2*(F_e.T*F_e - I)                         # Difference tensor
C = F_e.T * F_e
C_bar = F_e_bar.T*F_e_bar

'''Constants'''
#region
El = df.default_scalar_type(1.0e4)
nu = df.default_scalar_type(0.3)
mu = df.fem.Constant(mesh, El / (2 * (1 + nu)))
# mu = df.fem.Constant(mesh, 1.)
lmbda = df.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))
#endregion

# Compressible Neo-Hookean
kappa = 1e4
# psi_inc  = psi.neohookean(mu/2, C_bar)
psi_inc  = psi.neohookean(mu/2, C_bar)
psi_comp = psi.comp2(kappa, J) 
psi_=  psi_inc + psi_comp
P = ufl.diff(psi_, F_e)

weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions And Set Up Solver'''
P_with_bcs = weak_form
bcs, boundary_points = ddf.dirichlet_injection(mesh, bc_values, function_space)
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
        breakpoint()
        xdmf.write_meshtags(tags, mesh.geometry)

breakpoint()
print("Creating problem")
problem = NonlinearProblem(P_with_bcs, u, bcs)
print("Creating solver")
solver = NewtonSolver(mesh.comm, problem)
print("Solving")
solver.solve(u)
# solver.max_it = 500
print("Done solving")

# F_g_diff_function, F_g_diff_expression = ddf.to_tensor_map(F_g - F_g_old, mesh)
us = []; strain_funcs = []; strain_funcs_00 = []; F_gs = []; F_es = []; Fs = []

u_new = u.copy()
us.append(u_new)

'''Solve The Problem'''
for i in np.arange(8):

    if (i-1)%8 == 0:
        print(ddf.eval_expression(F_e, mesh))
        print(i)

    strain_function.interpolate(strain_expression)
    F_g_function.interpolate(F_g_expression)
    F_e_function.interpolate(F_e_expression)
    F_function.interpolate(F_expression)
    strain_function_00.interpolate(strain_expression_00)

    strain_function_new = strain_function.copy()
    strain_funcs.append(strain_function_new)
    strain_function_00_new = strain_function_00.copy()
    strain_funcs_00.append(strain_function_00_new)
    
    new_F_g = F_g_function.copy()
    new_F_e = F_e_function.copy()
    new_F = F_function.copy()
    F_gs.append(new_F_g)

    F_es.append(new_F_e)
    Fs.append(new_F)

    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)

F_g_diffs = []
for x, y in zip(F_gs, F_gs[1:]):
    F_g_h_function, F_g_h_expression = ddf.to_tensor_map(y - x, mesh)
    F_g_h_function.interpolate(F_g_h_expression)
    F_g_diffs.append(F_g_h_function)

F_h_s = []
for x, y in zip(Fs, Fs[1:]):
    F_h_function, F_h_expression = ddf.to_tensor_map(y - x, mesh)
    F_h_function.interpolate(F_h_expression)
    F_h_s.append(F_h_function)

pp.write_vector_to_paraview("ParaViewData/simple_growth.xdmf", mesh, us)
pp.write_tensor_to_paraview("ParaViewData/drivers.xdmf", mesh, strain_funcs)
pp.write_scalar_to_paraview("ParaViewData/driver_component.xdmf", mesh, strain_funcs_00)
pp.write_tensor_to_paraview("ParaViewData/F_g.xdmf", mesh, F_gs)
pp.write_tensor_to_paraview("ParaViewData/F_e.xdmf", mesh, F_es)
pp.write_tensor_to_paraview("ParaViewData/F.xdmf", mesh, Fs)
pp.write_tensor_to_paraview("ParaViewData/F_g_diffs.xdmf", mesh, F_g_diffs)
pp.write_tensor_to_paraview("ParaViewData/F_h_s.xdmf", mesh, F_h_s)