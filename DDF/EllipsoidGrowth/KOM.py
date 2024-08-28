import numpy as np 
from mpi4py import MPI
import ufl
import dolfinx
import dolfinx.log
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from pathlib import Path
from enum import Enum
from dolfinx.io import XDMFFile
from petsc4py import PETSc
import basix


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

f0  = geometry.f0 / ufl.sqrt(ufl.inner(geometry.f0, geometry.f0))
s0  = geometry.s0 / ufl.sqrt(ufl.inner(geometry.s0, geometry.s0))
n0  = geometry.n0 / ufl.sqrt(ufl.inner(geometry.n0, geometry.n0))

# dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

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
endo_neumann = ("ENDO", dolfinx.fem.Constant(geometry.mesh, 0.0))
# epi_neumann = "EPI", dolfinx.fem.Constant(geometry.mesh, 1.0))
neumann_bc_values = [endo_neumann]

'''Robin boundary conditions'''
# epi_robin = ("EPI", dolfinx.fem.Constant(geometry.mesh, 1.0), dolfinx.fem.Constant(geometry.mesh, 1.0))
base_robin = ("BASE", dolfinx.fem.Constant(geometry.mesh, 1.0), dolfinx.fem.Constant(geometry.mesh, 1.0))
robin_bc_values = []

natural_bcs = neumann_bc_values + robin_bc_values

'''Initiate first growth tensor'''
scalar_space = dolfinx.fem.functionspace(geometry.mesh, ("Lagrange", 2, (1, )))
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
#dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.variable(ufl.det(ufl.grad(X)))*ufl.dx(metadata={"quadrature_degree": 8})))
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
    Ecc = ufl.inner(E * geometry.s0, geometry.s0)
    Ecr = ufl.inner(E * geometry.s0, geometry.n0)
    Err = ufl.inner(E * geometry.n0, geometry.n0)
    
    return 0.5 * (Ecc + Err) + ufl.sqrt(0.25 * (Ecc - Err)**2 + Ecr**2)
    # return (E[1,1] + E[2,2])/2 + ufl.sqrt(((E[1,1] - E[2,2])/2)**2 + (E[1,2]*E[2,1]))
    # return E[1,1]
E_e_function = dolfinx.fem.Function(tensor_space)

dt = 0.1

F_g00 = ufl.inner(F_g_tot_function * geometry.f0, geometry.f0)
F_g11 = ufl.inner(F_g_tot_function * geometry.s0, geometry.s0)
F_g22 = ufl.inner(F_g_tot_function * geometry.n0, geometry.n0)
F_g1122 = 0.5*(F_g11 + F_g22)

t=dolfinx.fem.Constant(geometry.mesh, 0.0)

F_gff = ufl.conditional(ufl.ge(t, 0),
                        ufl.conditional(ufl.ge(ufl.inner(E_e_function * geometry.f0, geometry.f0), 0), 
                        k_growth(F_g00, f_l_slope, F_ff50)*f_ff_max*dt/(1 + ufl.exp(-f_f*(ufl.inner(E_e_function * geometry.f0, geometry.f0) - s_l50))) + 1, 
                        -f_ff_max*dt/(1 + ufl.exp(f_f*(ufl.inner(E_e_function * geometry.f0, geometry.f0) + s_l50))) + 1),
                        1)

F_gcc =  ufl.conditional(ufl.ge(t, 0),
                        ufl.conditional(ufl.ge(alg_max_princ_strain(E_e_function), 0), 
                        ufl.sqrt(k_growth(F_g1122, c_th_slope, F_cc50)*f_cc_max*dt/(1 + ufl.exp(-c_f*(alg_max_princ_strain(E_e_function) - s_t50))) + 1), 
                        ufl.sqrt(-f_cc_max*dt/(1 + ufl.exp(c_f*(alg_max_princ_strain(E_e_function) + s_t50))) + 1)),
                        1)
                        


F_g_I = ufl.as_tensor((
    (F_gff, 0, 0),
    (0, F_gcc, 0),
    (0, 0, F_gcc)))

R = ufl.as_tensor((
    (f0[0], s0[0], n0[0]),
    (f0[1], s0[1], n0[1]),
    (f0[2], s0[2], n0[2])
))

F_g = R.T*F_g_I*R

# F_g = F_gff * ufl.outer(f0, f0) + F_gcc * (ufl.Identity(3) - ufl.outer(f0, f0))
# F_g2 = F_gff * ufl.outer(f0, f0) + F_gcc * ufl.outer(s0, s0) + F_gcc * ufl.outer(n0, n0)
F_e = ufl.variable(F*ufl.inv(F_g))

F_g_tot = dolfinx.fem.Expression(F_g*F_g_tot_function, tensor_space.element.interpolation_points())
# Right Cauchy-Green tensor
E_e_expression = dolfinx.fem.Expression(0.5*(F_e.T*F_e - ufl.Identity(3)), tensor_space.element.interpolation_points())

J = ufl.variable(ufl.det(F_e))

F_e_bar = F_e#*pow(J, -1/3)
C_bar = F_e_bar.T*F_e_bar

'''Constants'''
mu = dolfinx.default_scalar_type(15)
kappa = dolfinx.default_scalar_type(1e2)

'''Create compressible strain energy function'''
# psi_inc  = psi.holzapfel(F_e, ufl.as_vector([1.0, 0.0, 0.0]))
psi_inc  = psi.neohookean(mu/2, C_bar)
psi_comp = psi.comp2(kappa, J) 
psi_=  psi_inc + psi_comp
P = ufl.diff(psi_, F)

'''Create weak form'''
weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions'''
if bc_values:
    print("applying dirichlet boundary conditions")
    bcs = ddf.dirichlet_injection_ellipsoid(geometry, bc_values, function_space)
else:
    print("no dirichlet boundary conditions applied")
    bcs = []

if natural_bcs:
    print("applying natural boundary conditions")
    neumann_bcs = ddf.neumann_injection_ellipsoid(geometry, neumann_bc_values, F_e, v)
    robin_bcs = ddf.robin_injection_ellipsoid(geometry, robin_bc_values, F_e, v, u)

    for bc in neumann_bcs + robin_bcs:
        weak_form += bc

else:
    print("no natural boundary conditions applied")
    

'''Assemble FEniCS solver'''
problem = NonlinearProblem(weak_form, u, bcs)
solver = NewtonSolver(geometry.mesh.comm, problem)

us = []
'''Solve once to get set point values etc.'''

print("solving")
solver.solve(u)
u_new = u.copy()
us.append(u_new)

F_g_f_tot = []; F_g_c_tot = []; F_e_list = []; J_e = []; J_g = []; J_g_tot = []; J_tot = []
'''Solve The Problem'''
N = 64
for i in range(N):

    if i < 10:
        endo_neumann[1].value += 4/10
        print("endo = ", endo_neumann[1].value)
    else:
        t.value = 1
    
        print("t.value = ", t.value)
        F_g_tot_function.interpolate(F_g_tot)
        E_e_function.interpolate(E_e_expression)

        if i % 1 == 0:       
            print("E_e = ", ddf.eval_expression(E_e_function, geometry.mesh))
            print("F_g = ", ddf.eval_expression(F_g, geometry.mesh))
            print("F_g_tot = ", ddf.eval_expression(F_g_tot_function, geometry.mesh))
            print("F_e = ", ddf.eval_expression(F_e, geometry.mesh))
            print("Step ", i)


    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)
    
    F_g_f_tot.append(ddf.eval_expression(F_g_tot_function[0,0], geometry.mesh)[0,0])
    F_g_c_tot.append(ddf.eval_expression(F_g_tot_function[1,1], geometry.mesh)[0,0])
    F_e_list.append(ddf.eval_expression(F_e[0,0], geometry.mesh)[0,0])
    J_e.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form((ufl.det(F_e))*ufl.dx(metadata={"quadrature_degree": 8}))))
    J_g.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form((ufl.det(F_g))*ufl.dx(metadata={"quadrature_degree": 8}))))
    J_tot.append(dolfinx.fem.assemble_scalar(dolfinx.fem.form((ufl.det(F))*ufl.dx(metadata={"quadrature_degree": 8}))))

lists_to_write = {
    "J_e": J_e,
    "J_g": J_g,
    "J_tot": J_tot
}

pp.write_lists_to_file("simulation_results.txt", lists_to_write)
pp.write_vector_to_paraview("../ParaViewData/simple_growth_ellipsoid.xdmf", geometry.mesh, us)
