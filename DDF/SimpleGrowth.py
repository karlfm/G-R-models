import numpy as np 
import ufl
import dolfinx as df
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from pathlib import Path
from dolfinx.io import XDMFFile
from petsc4py import PETSc
import basix

sys.path.insert(1, '/MyProject/MyCode/DDF/Geometry')
sys.path.insert(1, '/MyProject/MyCode/DDF/SpatialSolver')
sys.path.insert(1, '/MyProject/MyCode/DDF/PostProcessing')

import geometry as geo
import solver as solv 
import postprocessing as pp

#region
class bc:
    def __init__(self, bc_type, bc_condition):
        self.bc_type = bc_type
        self.bc_condition = bc_condition
    
    def info(self):
        return (self.bc_type, self.bc_condition)


'''Create Geometry'''
x_min, x_max, Nx = 0, 1, 4
y_min, y_max, Ny = 0, 1, 16
z_min, z_max, Nz = 0, 1, 16
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

'''Get Functions'''
function_space, u, v = solv.get_functions(mesh)

'''Create And Set Boundary Conditions'''
dirichlet_condition = lambda x : [x[0]*0, x[1]*0, x[2]*0]
neumann_condition_y = df.fem.Constant(mesh, df.default_scalar_type((0, 0, 1)))
neumann_condition_z = df.fem.Constant(mesh, df.default_scalar_type((0, 0, 1)))
neumann_condition = df.fem.Constant(mesh, df.default_scalar_type((0, 0, 1)))
no_bc, dirichlet, neumann, robin = 0, 1, 2, 3
x_left  = bc(dirichlet, lambda x : [x[0]*0, x[1]*0, x[2]*0])
x_right = bc(dirichlet, lambda x : [x[0]*1, x[1]*0, x[2]*0])
y_left  = bc(no_bc, 0)
y_right = bc(no_bc, 0)
z_left  = bc(no_bc, 0)
z_right = bc(no_bc, 0)

bc_values = [
    (x_left.info()), (x_right.info()),   # x-axis boundary type and value
    (y_left.info()), (y_right.info()),   # y-axis boundary type and value 
    (z_left.info()), (z_right.info())    # z-axis boundary type and value
    ]
#endregion

infinitesimal_strain = 1/2*(ufl.grad(u).T * ufl.grad(u) - ufl.Identity(len(u)))
strain_function, strain_expression = solv.to_tensor_map(infinitesimal_strain, mesh)

F_g = ufl.as_tensor((
    (1 + strain_function[0, 0], 0, 0),
    (0, 1, 0),
    (0, 0, 1)))

'''Create Stress Tensor'''
invF_G = ufl.inv(F_g)

I = ufl.Identity(len(u))                        # Identity tensor
F = ufl.variable(I + ufl.grad(u))               # Deformation tensor
F_e = ufl.variable(F*invF_G)                    # Elasticity tensor
E = 1/2*(F_e.T*F_e - I)                         # Curvature tensor / difference tensor

J = ufl.det(F_e)                                # Determinant
C = ufl.variable(F_e.T * F_e)                   # Metric tensor / ratio tensor

Ic = ufl.variable(ufl.tr(C))                    # First invariant

El = df.default_scalar_type(1.0e4)
nu = df.default_scalar_type(0.3)
mu = df.fem.Constant(mesh, El / (2 * (1 + nu)))
lmbda = df.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))

#Compressible Neo-Hookean
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
P = ufl.diff(psi, F_e)
sigma = ufl.inner(ufl.grad(v), P) * ufl.dx 

von_Mises = ufl.sqrt(
        (P[0, 0] - P[1, 1])**2 + (P[1, 1] - P[2, 2])**2 + 
        (P[2, 2] - P[0, 0])**2 + 6*(P[0, 1]**2 + P[1, 2]**2 + P[2, 0]**2)
    )
function, expression = solv.to_scalar_map(von_Mises, mesh)

'''Apply Boundary Conditions And Set Up Solver'''
P_with_bcs, bcs = solv.apply_boundary_conditions(mesh, bc_values, sigma, function_space, v, X)
problem = NonlinearProblem(P_with_bcs, u, bcs)

solver = NewtonSolver(mesh.comm, problem)
solver.solve(u)

us = []
von_Mises_stresses = []
u_new = u.copy()
us.append(u_new)

'''Solve The Problem'''
for i in np.arange(16):
    print(i)
    strain_function.interpolate(strain_expression)
    function.interpolate(expression)
    solver.solve(u)

    u_new = u.copy()
    us.append(u_new)
    von_Mises = function.copy()
    print(solv.eval_expression(strain_function, mesh))
    von_Mises_stresses.append(von_Mises)

### WHAT DOES THIS DO?
mesh.topology.create_connectivity(
    mesh.topology.dim-1, mesh.topology.dim
    )

# with XDMFFile(mesh.comm, "Hookean_facet_tags.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(facet_tags, mesh.geometry)

# Not really sure about this
AA = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
BB = df.fem.FunctionSpace(mesh, AA)
u1 = df.fem.Function(BB)

filenameStress = Path("VonMissesStress.xdmf")
filenameStress.unlink(missing_ok=True)
filenameStress.with_suffix(".h5").unlink(missing_ok=True)
foutStress = df.io.XDMFFile(mesh.comm, filenameStress, "w")
foutStress.write_mesh(mesh)

pp.write_to_paraview("SimpleGrowth.xdmf", mesh, us)
pp.write_vm_to_paraview("VonMissesStress.xdmf", mesh, von_Mises_stresses)