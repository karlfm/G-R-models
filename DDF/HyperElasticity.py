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

x_min, x_max, Nx = 0, 1, 64
y_min, y_max, Ny = 0, 1, 16
z_min, z_max, Nz = 0, 1, 16
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

function_space, u, v = solv.get_functions(mesh)

I, F, J, C, E = solv.get_basic_tensors(u)

no_bc, dirichlet, neumann, robin = 0, 1, 2, 3

dirichlet_condition = lambda x : [x[0]*0, x[1]*0, x[2]*0]
x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)   
neumann_condition = df.fem.Constant(mesh, df.default_scalar_type((20000, 0, 0)))
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

# Elasticity parameters
Ic = ufl.variable(ufl.tr(C))

El = df.default_scalar_type(1.0e4)
nu = df.default_scalar_type(0.3)
mu = df.fem.Constant(mesh, El / (2 * (1 + nu)))
lmbda = df.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# Stress
# Hyper-elasticity


'''STRAIN TENSORS'''

P = ufl.diff(psi, F)
sigma = ufl.inner(ufl.grad(v), P) * ufl.dx 
facet_tags = solv.set_boundary_types(mesh, boundary_types, X)


P_with_bcs, bcs = solv.apply_boundary_conditions(mesh, facet_tags, bc_values, sigma, function_space, v, F, n)

problem = NonlinearProblem(P_with_bcs, u, bcs)
solver = NewtonSolver(mesh.comm, problem)
# Set Newton solver options

solver.solve(u)


#################################################################################

### WHAT DOES THIS DO?
mesh.topology.create_connectivity(
    mesh.topology.dim-1, mesh.topology.dim
    )

with XDMFFile(mesh.comm, "facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tags, mesh.geometry)

# Not really sure about this
AA = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
BB = df.fem.FunctionSpace(mesh, AA)
u1 = df.fem.Function(BB)

filename = Path("myDisplacement.xdmf")
filename.unlink(missing_ok=True)
filename.with_suffix(".h5").unlink(missing_ok=True)
fout = df.io.XDMFFile(mesh.comm, filename, "w")
fout.write_mesh(mesh)

u1.interpolate(u)
fout.write_function(u1, 1)
