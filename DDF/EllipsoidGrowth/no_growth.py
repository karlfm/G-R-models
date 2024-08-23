import numpy as np 
from mpi4py import MPI
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
sys.path.insert(1, '/home/shared/')

import ddf as ddf 
import postprocessing as pp
import hyperelastic_models as psi

import cardiac_geometries
import cardiac_geometries.geometry

geodir = Path("lv_ellipsoid")
if not geodir.exists():
    cardiac_geometries.mesh.lv_ellipsoid(outdir=geodir, create_fibers=True, fiber_space="P_2")

geometry = cardiac_geometries.geometry.Geometry.from_folder(
    comm=MPI.COMM_WORLD,
    folder=geodir,
)

with dolfinx.io.VTXWriter(geometry.mesh.comm, "fibers.bp", [geometry.f0, geometry.s0, geometry.n0], engine="BP4") as vtx:
    vtx.write(0.0)

'''Get Functions'''
function_space, u, v = ddf.get_functions(geometry.mesh)

# bc_location, component, value
base_x  = ("BASE", 0, dolfinx.default_scalar_type(0))
base_y  = ("BASE", 1, dolfinx.default_scalar_type(0))
base_z  = ("BASE", 2, dolfinx.default_scalar_type(0))
endo_x  = ("ENDO", 0, dolfinx.default_scalar_type(0))
endo_y  = ("ENDO", 1, dolfinx.default_scalar_type(0))
endo_z  = ("ENDO", 2, dolfinx.default_scalar_type(0))
epi_x   = ("EPI" , 0, dolfinx.default_scalar_type(0))
epi_y   = ("EPI" , 1, dolfinx.default_scalar_type(0))
epi_z   = ("EPI" , 2, dolfinx.default_scalar_type(0))
bc_values = [base_x, base_y, base_z]#, base_y, base_z]#, endo_x, endo_y]#, endo_z, epi_x, epi_y, epi_z]

'''Neumann boundary conditions'''
endo_neumann = ("ENDO", dolfinx.fem.Constant(geometry.mesh, 1.0))
neumann_bc_values = [endo_neumann]

'''Robin boundary conditions'''
epi_robin = ("EPI", 1, dolfinx.fem.Constant(geometry.mesh, 1.0), dolfinx.fem.Constant(geometry.mesh, 0.5))
robin_bc_values = [epi_robin]

'''Kinematics'''
I = ufl.Identity(len(u))
F = ufl.variable(I + ufl.grad(u))
J = ufl.variable(ufl.det(F))

F_bar = F#*pow(J, -1/3)
C_bar = F_bar.T*F_bar

'''Create compressible strain energy function'''
psi_inc  = psi.holzapfel(F, geometry.f0)
psi_=  psi_inc
P = ufl.diff(psi_, F)

'''Create weak form'''
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions'''
bcs = ddf.dirichlet_injection_ellipsoid(geometry, bc_values, function_space)
neumann_bcs = ddf.neumann_injection_ellipsoid(geometry, neumann_bc_values, F, v)
robin_bcs = ddf.robin_injection_ellipsoid(geometry, robin_bc_values, F, v, u)

for bc in robin_bcs + neumann_bcs:
    weak_form += bc

'''Assemble FEniCS solver'''
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(geometry.mesh.comm, problem)

us = []
'''Solve once to get set point values etc.'''

print("solving")
solver.solve(u)
u_new = u.copy()
us.append(u_new)
F_g_f_tot = []; F_g_c_tot = []; F_e_list = []

'''Solve The Problem'''
N = 1
for i in range(N):

    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)

pp.write_vector_to_paraview("../ParaViewData/simple_growth.xdmf", geometry.mesh, us)