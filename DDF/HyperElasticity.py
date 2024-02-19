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

x_min, x_max, Nx = 0, 1, 16
y_min, y_max, Ny = 0, 1, 8
z_min, z_max, Nz = 0, 1, 8
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

function_space, u, v = solv.get_functions(mesh)



no_bc, dirichlet, neumann, robin = 0, 1, 2, 3

dirichlet_condition = lambda x : [x[0]*0, x[1]*0, x[2]*0]
x = ufl.SpatialCoordinate(mesh)
n = ufl.FacetNormal(mesh)   
neumann_condition = df.fem.Constant(mesh, df.default_scalar_type((0, 0, 50)))
boundary_types = [
    (dirichlet, dirichlet), 
    (no_bc, no_bc), 
    (no_bc, no_bc)
    ]

bc_values = [
    (dirichlet, lambda x : [x[0]*0, x[1]*0, x[2]*0]), (dirichlet, lambda x : [x[0]*1, x[1]*0, x[2]*0]),     # x-axis boundary type and value
    (no_bc, 0), (no_bc, 0),   # y-axis boundary type and value 
    (no_bc, 0), (no_bc, 0)    # z-axis boundary type and value
    ]

facet_tags = solv.set_boundary_types(mesh, boundary_types, X)
# lambdaF = lambda u : ufl.variable(I + ufl.grad(u))             # Deformation tensor
# F = lambdaF(u)

# Elasticity parameters




# Strain energy function
t = df.fem.Constant(mesh, df.default_scalar_type(0))
F_g = ufl.as_tensor(((1+t, 0, 0), (0, 1, 0), (0, 0, 1)))
invF_G = ufl.inv(F_g)
#F_e = ufl.variable(ufl.dot(F))
I = ufl.Identity(len(u))                      # Identity tensor
F = ufl.variable(I + ufl.grad(u))
F_e = F*invF_G                                  # Deformation tensor
# F_e = F
E = 1/2*(F_e.T*F_e - I)                           # Curvature tensor / difference tensor
J = ufl.det(F_e)                                # Determinant
C = ufl.variable(F_e.T * F_e)                     # Metric tensor / ratio tensor
Ic = ufl.variable(ufl.tr(C))                  # First invariant
El = df.default_scalar_type(1.0e4)
nu = df.default_scalar_type(0.3)
mu = df.fem.Constant(mesh, El / (2 * (1 + nu)))
lmbda = df.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))

psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
P = ufl.diff(psi, F)
sigma = ufl.inner(ufl.grad(v), P) * ufl.dx 
P_with_bcs, bcs = solv.apply_boundary_conditions(mesh, facet_tags, bc_values, sigma, function_space, v)
problem = NonlinearProblem(P_with_bcs, u, bcs)
solver = NewtonSolver(mesh.comm, problem)

solver.solve(u)


solver = NewtonSolver(mesh.comm, problem)
solver.solve(u)
us = []
for i in np.linspace(0, 1, 10, endpoint=True):
    print(i)
    t.value = i
    print(solv.eval_expression(F_e, [0.5, 0.5, 0.5], mesh))
    # F_e = F*ufl.inv(F_g(t))
    solver.solve(u)
    u_new = u.copy() 
    us.append(u_new)
    # F = F_e*F_g(t)
    # F_e = F*ufl.inv(F_g(t))

# '''STRAIN TENSORS'''

# Set Newton solver options

# F_e = (lambdaF(u))

# breakpoint()
# for t in np.linspace(0, 0.5, 1, endpoint=True):
#     print("t = ", t)
#     # F_e = ufl.variable(F*ufl.inv(F_g(t)))

#     P = ufl.diff(psi, F_e)
#     breakpoint()
#     # P = ufl.diff(psi, F)

#     sigma = ufl.inner(ufl.grad(v), P) * ufl.dx     
#     P_with_bcs, bcs = solv.apply_boundary_conditions(mesh, facet_tags, bc_values, sigma, function_space, v)
#     problem = NonlinearProblem(P_with_bcs, u, bcs)
#     solver = NewtonSolver(mesh.comm, problem)
#     solver.solve(u)
#     u_new = u.copy()  
#     F_e_old = F_e 
#     #vms.append(von_mis(F_e_old)) 
#     us.append(u_new)
#     F = F_e*F_g(t)


#################################################################################

### WHAT DOES THIS DO?
mesh.topology.create_connectivity(
    mesh.topology.dim-1, mesh.topology.dim
    )

with XDMFFile(mesh.comm, "Hookean_facet_tags.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(facet_tags, mesh.geometry)

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