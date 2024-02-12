import numpy as np 
import ufl
import dolfinx as df
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from pathlib import Path
from dolfinx.io import XDMFFile

sys.path.insert(1, '/MyProject/MyCode/DDF/Geometry')
sys.path.insert(1, '/MyProject/MyCode/DDF/SpatialSolver')

import geometry as geo
import solver as solv 

xs = 8
ys = 8
zs = 8

mesh = geo.create_unit_cube(xs, ys, zs)

function_space, u, v = solv.get_functions(mesh)

I, F, J, C, E = solv.get_basic_tensors(u)

no_bc, dirichlet, neumann, robin = 0, 1, 2, 3


def sl(expr, points):
    maxVal = 0
    for point in points:
        currentPoint = eval_expression(expr, point)
        # print(currentPoint)
        if currentPoint[0][0] > maxVal:
            # print(currentPoint[0][0])
            maxVal = currentPoint[0][0]
            # print("maxVal = ", maxVal)
    return maxVal - setPoint

def st(expr, points):
    minVal = 0
    for point in points:
        currentPoint = solv.eval_expression(expr, point)
        mini = max(
            [currentPoint[0][4], currentPoint[0][5], currentPoint[0][7], currentPoint[0][8]]
            )
        if mini < minVal:
            minVal = mini
    return minVal - setPoint

sl_val = 0
st_val = 0

'''CONSTANTS'''
C_pas = 0.44
b_f = 18.5
b_t = 3.58
b_fr = 1.63
C_comp = 350

f_ffmax = 0.3
f_f = 150
s_l50 = 0.06
F_ff50 = 2.35
f_lengthslope = 40
f_ccmax = 0.1
c_f = 75
s_t50 = 0.07
F_cc50 = 1.28
c_thicknessslope = 60

E_ffset = 1     #THE PAPER DOESNT SPECIFY

'''MATH'''
I = ufl.Identity(3)                                 # Identity tensor
F = ufl.variable(I + ufl.grad(u))                   # Deformation tensor
currentF = F

V_f = df.fem.FunctionSpace(mesh, ufl.TensorElement("DG", mesh.ufl_cell(), 0))
F_gff_cum = df.fem.Function(V_f)
F_gcc_cum = df.fem.Function(V_f)
F_gff_cum.x.array[:] = 1
F_gcc_cum.x.array[:] = 1

k_ff = 1 / (1 + ufl.exp(f_lengthslope*(currentF[0,0] - F_ff50)))
k_cc = 1 / (1 + ufl.exp(c_thicknessslope*(currentF[1,1] - F_cc50)))

F_gff = ufl.conditional(
    ufl.ge(sl_val,0),
    k_ff*f_ffmax/(1 + ufl.exp(-f_f*(sl_val - s_l50))) + 1,
    -f_ffmax/(1 + ufl.exp(f_f*(sl_val + s_l50))) + 1)

F_gcc = ufl.conditional(
    ufl.ge(st_val,0), 
    ufl.sqrt(k_cc*f_ccmax/(1 + ufl.exp(-c_f*(st_val - s_t50))) + 1), 
    ufl.sqrt(-f_ccmax/(1 + ufl.exp(c_f*(st_val - s_t50))) + 1))

F_g = ufl.as_tensor((
    (F_gff, 0, 0),
    (0, F_gcc, 0),
    (0, 0, F_gcc)))

F_e = F * ufl.inv(F_g)

newE = lambda F : 1/2*(F.T*F - I)

E = 1/2*(F_e.T*F_e - I)    # Green-Lagrange tensor (difference tensor)

'''Strain Energy Density Function'''
Q = (
    b_f*E[0, 0]**2 
    + b_t*(E[1, 1]**2 + E[2, 2]**2 + 2*E[1, 2]*E[2, 1]) 
    + b_fr*(2*E[0, 1]*E[1, 0] + 2*E[0, 2]*E[2, 0])
)

E_cross = ufl.as_tensor(((E[1, 1], E[1, 2]), (E[2, 1], E[2, 2])))

'''STRAIN TENSORS'''

J = ufl.det(F_e)             # Determinant of deformation tensor
C = pow(J, -float(2 / 3)) * F_e.T * F_e       # Cauchy-Green tensor (ratio tensor)

W_pas = 1/2*C_pas*(ufl.exp(Q - 1))
W_comp = 1/2*C_comp*(J - 1)*ufl.ln(J)

W = W_pas + W_comp

P = ufl.diff(W, F)
sigma = ufl.inner(ufl.grad(v), P) * ufl.dx 

'''BOUNDARY CONDITIONS'''
dirichlet_condition = lambda x : [x[0]*0, x[1]*0, x[2]*0]
x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)   
neumann_condition = df.fem.Constant(mesh, df.default_scalar_type((1.5, 0, 0)))
boundary_types = [
    (dirichlet, neumann), 
    (no_bc, no_bc), 
    (no_bc, no_bc)
    ]

bc_values = [
    (dirichlet, dirichlet_condition), (neumann, neumann_condition),     # x-axis boundary type and value
    (no_bc, 0), (no_bc, 0),   # y-axis boundary type and value 
    (no_bc, 0), (no_bc, 0)    # z-axis boundary type and value
    ]


P_with_bcs, bcs = solv.apply_boundary_conditions(mesh, boundary_types, bc_values, sigma, function_space, u, v)

problem = NonlinearProblem(P_with_bcs, u, bcs)
solver = NewtonSolver(mesh.comm, problem)
# Set Newton solver options
# solver.atol = 1e-8
# solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

solver.solve(u)

#################################################################################
facet_tags = solv.set_boundary_types(mesh, boundary_types)

### WHAT DOES THIS DO?
mesh.topology.create_connectivity(
    mesh.topology.dim-1, mesh.topology.dim
    )

with XDMFFile(mesh.comm, "Kerchoff_facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tags, mesh.geometry)

# Not really sure about this
AA = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
BB = df.fem.FunctionSpace(mesh, AA)
u1 = df.fem.Function(BB)

filename = Path("myKerchoffDisplacement.xdmf")
filename.unlink(missing_ok=True)
filename.with_suffix(".h5").unlink(missing_ok=True)
fout = df.io.XDMFFile(mesh.comm, filename, "w")
fout.write_mesh(mesh)

u1.interpolate(u)
fout.write_function(u1, 1)
