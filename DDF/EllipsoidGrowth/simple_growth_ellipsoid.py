import numpy as np 
from mpi4py import MPI
import ufl
import dolfinx
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


sys.path.insert(1, '/home/shared/helper')
sys.path.insert(1, '/home/shared/')

# import geometry as geo
import ddf as ddf 
import postprocessing as pp
import hyperelastic_models as psi
import growth_laws

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
bc_values = [base_x]#, base_y, base_z]#, base_y, base_z]#, endo_x, endo_y]#, endo_z, epi_x, epi_y, epi_z]

'''Neumann boundary conditions'''
neumann_bc = dolfinx.fem.Constant(geometry.mesh, 2.0)
endo_neumann = ("ENDO", neumann_bc)
neumann_bc_values = [endo_neumann]

'''Robin boundary conditions'''
epi_robin = ("EPI", 1, dolfinx.fem.Constant(geometry.mesh, 1.0), dolfinx.fem.Constant(geometry.mesh, 0.5))
robin_bc_values = [epi_robin]

'''Initiate first growth tensor'''
tensor_space = dolfinx.fem.functionspace(geometry.mesh, basix.ufl.element(family="CG", cell=str(geometry.mesh.ufl_cell()), degree=2, shape=(3,3)))
X = ufl.SpatialCoordinate(geometry.mesh)       # get Identity without it being a constant
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

t=dolfinx.fem.Constant(geometry.mesh, 0.0)

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

M = ufl.outer(geometry.f0, geometry.f0) + ufl.outer(geometry.s0, geometry.s0) + ufl.outer(geometry.n0, geometry.n0)

F_g_Euclid = ufl.as_tensor((
    (F_gff, 0, 0),
    (0, F_gcc, 0),
    (0, 0, F_gcc)))

# F_g = M*F_g_Euclid*M.T
F_g = F_g_Euclid

F_e = ufl.variable(F*ufl.inv(F_g))

F_g_tot = dolfinx.fem.Expression(F_g*F_g_tot_function, tensor_space.element.interpolation_points())

s_expression = dolfinx.fem.Expression(0.5*(F_e.T*F_e - ufl.Identity(3)), tensor_space.element.interpolation_points())

J = ufl.variable(ufl.det(F_e))

F_e_bar = F_e*pow(J, -1/3)
C_bar = F_e_bar.T*F_e_bar

'''Constants'''
El = dolfinx.default_scalar_type(1.0e4)
nu = dolfinx.default_scalar_type(0.3)
mu = dolfinx.fem.Constant(geometry.mesh, El / (2 * (1 + nu)))
lmbda = dolfinx.fem.Constant(geometry.mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))
kappa = 1e6

'''Create compressible strain energy function'''
psi_inc  = psi.neohookean(1, C_bar)
psi_comp = psi.comp2(kappa, J) 
psi_=  psi_inc + psi_comp
P = ufl.diff(psi_, F)

'''Create weak form'''
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions'''
bcs = ddf.dirichlet_injection_ellipsoid(geometry, bc_values, function_space)
neumann_bcs = ddf.neumann_injection_ellipsoid(geometry, neumann_bc_values, F_e, v)
robin_bcs = ddf.robin_injection_ellipsoid(geometry, robin_bc_values, F_e, v, u)

for bc in neumann_bcs + robin_bcs:
    weak_form += bc

'''Assemble FEniCS solver'''
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(geometry.mesh.comm, problem)

us = []
'''Solve once to get set point values etc.'''

# ddf.eval_expression(F_g, geometry.mesh)
solver.solve(u)
u_new = u.copy()
us.append(u_new)
F_g_f_tot = []; F_g_c_tot = []; F_e_list = []

'''Solve The Problem'''
N = 32
for i in range(N):

    t.value = i
 
    F_g_tot_function.interpolate(F_g_tot)
    s_function.interpolate(s_expression)

    # breakpoint()
    if i % 1 == 0:       
        print("Step ", i)
        print("s = ", ddf.eval_expression(s_function, geometry.mesh))
        print("F_g = ", ddf.eval_expression(F_g, geometry.mesh))
        print("F_g_tot = ", ddf.eval_expression(F_g_tot_function, geometry.mesh))
        print("F_e = ", ddf.eval_expression(F_e, geometry.mesh))

    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)

    F_g_f_tot.append(ddf.eval_expression(F_g_tot_function[0,0], geometry.mesh)[0,0])
    F_g_c_tot.append(ddf.eval_expression(F_g_tot_function[1,1], geometry.mesh)[0,0])
    F_e_list.append(ddf.eval_expression(F_e[0,0], geometry.mesh)[0,0])



pp.write_vector_to_paraview("ParaViewData/simple_growth_ellipsoid.xdmf", geometry.mesh, us)