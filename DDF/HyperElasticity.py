import numpy as np 
import ufl
import dolfinx as df
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from pathlib import Path
from dolfinx.io import XDMFFile
from petsc4py import PETSc


sys.path.insert(1, '/MyProject/MyCode/DDF/Geometry')
sys.path.insert(1, '/MyProject/MyCode/DDF/SpatialSolver')

import geometry as geo
import solver as solv 
import postprocessing as pp

def von_mis(F_e):


    # s = sigma(F_e) - 1. / 3 * ufl.tr(sigma(F_e)) * ufl.Identity(len(u))
    # von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s,s))
    sig = sigma(F_e)
    von_Mises = (
        1/2 * ((sig[0 ,0] - sig[1, 1])**2 + 
               (sig[1, 1] - sig[2, 2])**2 + 
               (sig[2, 2] - sig[0, 0])**2 + 
               6*(sig[1, 2]**2 + sig[2, 0]**2 + sig[0, 1]**2))
    )
    return von_Mises


class bc:
    def __init__(self, bc_type, bc_condition):
        self.bc_type = bc_type
        self.bc_condition = bc_condition
    
    def info(self):
        return (self.bc_type, self.bc_condition)


'''Create Geometry'''
x_min, x_max, Nx = 0, 1, 16
y_min, y_max, Ny = 0, 1, 8
z_min, z_max, Nz = 0, 1, 8
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)


'''Get Functions'''
function_space, u, v = solv.get_functions(mesh)


'''Create And Set Boundary Conditions'''
dirichlet_condition = lambda x : [x[0]*0, x[1]*0, x[2]*0]
neumann_condition = df.fem.Constant(mesh, df.default_scalar_type((0, 0, 50)))
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


'''Create Stress Tensor'''
# V = df.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))

# Q = df.fem.FiniteElement(mesh, ("Quadrature", 4, (3,3)))
Q = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), 4, (3,3))
q = df.fem.FunctionSpace(mesh, Q)
F_g = df.fem.Function(q)

t = df.fem.Constant(mesh, df.default_scalar_type(0))
F_g = ufl.as_tensor(((1+t, 0, 0), (0, 1, 0), (0, 0, 1)))
invF_G = ufl.inv(F_g)

I = ufl.Identity(len(u))                        # Identity tensor
F = ufl.variable(I + ufl.grad(u))               # Deformation tensor
F_e = ufl.variable(F*invF_G)                    # Elasticity tensor
E = 1/2*(F_e.T*F_e - I)                         # Curvature tensor / difference tensor

# E_expr = df.fem.Expression(E, q.element.interpolation_points())

#t.interpolate(E)


J = ufl.det(F_e)                                # Determinant
C = ufl.variable(F_e.T * F_e)                   # Metric tensor / ratio tensor

Ic = ufl.variable(ufl.tr(C))                    # First invariant

El = df.default_scalar_type(1.0e4)
nu = df.default_scalar_type(0.3)
mu = df.fem.Constant(mesh, El / (2 * (1 + nu)))
lmbda = df.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))

psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
P = ufl.diff(psi, F_e)
sigma = ufl.inner(ufl.grad(v), P) * ufl.dx 


'''Apply Boundary Conditions And Set Up Solver'''
P_with_bcs, bcs = solv.apply_boundary_conditions(mesh, bc_values, sigma, function_space, v, X)
problem = NonlinearProblem(P_with_bcs, u, bcs)
solver = NewtonSolver(mesh.comm, problem)


'''Solve The Problem'''
us = []
for i in np.linspace(0, 1, 8, endpoint=True):
    print(i)

    solver.solve(u)
    localE = solv.eval_expression(E, [0.5, 0.5, 0.5], mesh)
    print(solv.eval_expression(E, [0.5, 0.5, 0.5], mesh))
    
    t.value = i

    #F_g += E
    # F_g.interpolate(E_expr)
    u_new = u.copy() 
    us.append(u_new)









##################################################################################################################

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

pp.write_to_paraview("NeuHookeanDisplacement.xdmf", mesh, us)
#pp.write_vm_to_paraview("VonMissesStress.xdmf", mesh, vms)