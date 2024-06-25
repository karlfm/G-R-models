import numpy as np 
import ufl
import dolfinx as df
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from pathlib import Path
from enum import Enum
from dolfinx.io import XDMFFile
from petsc4py import PETSc
import basix

import cardiac_geometries
import cardiac_geometries.geometry

from mpi4py import MPI

sys.path.insert(1, '/home/shared/helper')

# import geometry as geo
import ddf as ddf 
import postprocessing as pp
import hyperelastic_models as psi
import growth_laws

geodir = Path("lv_ellipsoid")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True)

geometry = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

'''Get Functions'''
function_space, u, v = ddf.get_functions(geometry.mesh)

# bc_location, component, value
base_x  = ("BASE", 0, df.default_scalar_type(0))
base_y  = ("BASE", 1, df.default_scalar_type(0))
base_z  = ("BASE", 2, df.default_scalar_type(0))
endo_x  = ("ENDO", 0, df.default_scalar_type(0))
endo_y  = ("ENDO", 1, df.default_scalar_type(0))
endo_z  = ("ENDO", 2, df.default_scalar_type(0))
epi_x   = ("EPI" , 0, df.default_scalar_type(0))
epi_y   = ("EPI" , 1, df.default_scalar_type(0))
epi_z   = ("EPI" , 2, df.default_scalar_type(0))
bc_values = [base_x]#, base_y, base_z]#, endo_x, endo_y]#, endo_z, epi_x, epi_y, epi_z]

'''Neumann boundary conditions'''
neumann_bc = df.fem.Constant(geometry.mesh, 400.0)
endo_neumann = ("ENDO", neumann_bc)
neumann_bc_values = [endo_neumann]

'''Robin boundary conditions'''
neumann_bc = df.fem.Constant(geometry.mesh, 400.0)
epi_robin = ("EPI", neumann_bc, df.fem.Constant(geometry.mesh, .5))
robin_bc_values = [epi_robin]

'''Create Stress Tensor'''
V = df.fem.functionspace(geometry.mesh, basix.ufl.element(family="DG", cell=str(geometry.mesh.ufl_cell()), degree=0, shape=(3,3)))

strain = 1/2*(ufl.grad(u).T + ufl.grad(u))# + ufl.grad(u).T*ufl.grad(u))
strain_function, strain_expression = ddf.to_tensor_map(strain, geometry.mesh)
X = ufl.SpatialCoordinate(geometry.mesh)     # get Identity without it being a constant
F_0 = ufl.grad(X)                   # --//--    
F_0_func = df.fem.Function(V)       # this gets updated in the loop
F0_form = df.fem.Expression(F_0, V.element.interpolation_points())  # 
F_0_func.interpolate(F0_form)

'''From the paper'''
I = ufl.Identity(len(u))
F = ufl.variable(I + ufl.grad(u))
F_g = growth_laws.F_g1(strain_function, F_0_func)*F_0_func
# F_g = growth_laws.KUR(1, 1, strain_function, geometry.mesh)
# F_g = growth_laws.F_g1(strain_function)
F_e = ufl.variable(F*ufl.inv(F_g))

F_g_function, F_g_expression = ddf.to_tensor_map(F_g, geometry.mesh)
# F_g_function, F_g_expression = ddf.to_tensor_map(F_g_new, mesh)
F_e_function, F_e_expression = ddf.to_tensor_map(F_e, geometry.mesh)
F_function, F_expression = ddf.to_tensor_map(F_e*F_g, geometry.mesh)

J = ufl.variable(ufl.det(F_e))

F_e_bar = F_e*pow(J, -1/3)
C_bar = F_e_bar.T*F_e_bar

'''Constants'''
El = df.default_scalar_type(1.0e4)
nu = df.default_scalar_type(0.3)
mu = df.fem.Constant(geometry.mesh, El / (2 * (1 + nu)))
lmbda = df.fem.Constant(geometry.mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))
kappa = 1e4

'''Create compressible strain energy function'''
psi_inc  = psi.neohookean(mu/2, C_bar)
psi_comp = psi.comp2(kappa, J) 
psi_=  psi_inc + psi_comp
P = ufl.diff(psi_, F_e)*F_e.T/J

'''Create weak form'''
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions'''
bcs = ddf.dirichlet_injection_ellipsoid(geometry, bc_values, function_space)
neumann_bcs = ddf.neumann_injection_ellipsoid(geometry, neumann_bc_values, weak_form, F_e, v)
robin_bcs = ddf.robin_injection_ellipsoid(geometry, robin_bc_values, weak_form, F_e, v, u)

for bc in neumann_bcs + robin_bcs:
    weak_form += bc

'''Assemble FEniCS solver'''
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(geometry.mesh.comm, problem)

us = []; strain_funcs = []; F_gs = []; F_es = []; Fs = []

u_new = u.copy()
us.append(u_new)

# I don't understand why this is needed
# I think it is because we need to initiate the first previous total growth step to create the growth accumulation tensor
strain_function.interpolate(strain_expression)
F_g_prev = growth_laws.F_g1(strain_function, F_0_func)

'''Solve The Problem'''
for i in np.arange(32):

    print(i)

    # print(ddf.eval_expression(pow((1/2*(ufl.sqrt(2*strain_function[1,1] + 1) - 1/2 - 1) + 1), 1/3), mesh))

    # F_g_new = ufl.dot(F_g_new, F_g)

    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)

    strain_function.interpolate(strain_expression)
    F_0_func.interpolate(df.fem.Expression(F_g_prev, V.element.interpolation_points()))


pp.write_vector_to_paraview("ParaViewData/simple_growth.xdmf", geometry.mesh, us)