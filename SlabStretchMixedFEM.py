import numpy as np
import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc

'''
TODO: REMOVE THIS
'''
def define_bcs(V, mesh, stretch_fun):

    coords = mesh.geometry.x
    xmin = min(coords[:, 0])
    xmax = max(coords[:, 0])
    ymin = min(coords[:, 1])
    zmin = min(coords[:, 2])

    xmin_bnd = lambda x : np.isclose(x[0], xmin)
    xmax_bnd = lambda x : np.isclose(x[0], xmax)
    ymin_bnd = lambda x : np.isclose(x[1], ymin)
    zmin_bnd = lambda x : np.isclose(x[2], zmin)

    fdim = 2

    u_dixed = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))

    bnd_funs = [xmin_bnd, ymin_bnd, zmin_bnd]
    components = [0, 1, 2]

    bcs = []

    for bnd_fun, comp in zip(bnd_funs, components):
        boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bnd_fun)
        dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)

        bc = dolfinx.fem.dirichletbc(u_dixed, dofs, V.sub(0))
        bcs.append(bc)

    # moving bc

    boundary_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, xmax_bnd)
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)

    bc = dolfinx.fem.dirichletbc(stretch_fun, dofs, V.sub(0))
    bcs.append(bc)

    return bcs

'''
CREATE MESH/FUNCTION SPACE
'''
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

mixedSpace = dolfinx.fem.FunctionSpace(mesh, P2 * P1)

functions = dolfinx.fem.Function(mixedSpace)

u, p = functions.split()
v, q = ufl.TestFunctions(mixedSpace)

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree" : 4})

'''
MATH
'''
I        = lambda u           : ufl.Identity(len(u))
F        = lambda u           : ufl.variable(I(u) + ufl.grad(u))
J        = lambda u           : ufl.det(F(u))
C        = lambda u           : pow(J(u), -float(2 / 3))*F(u).T * F(u)
Psi      = lambda u           : 2*(ufl.tr(C(u)) - 3)
P        = lambda u, p        : ufl.diff(Psi(u), F(u)) + p * J(u) * ufl.inv(F(u).T)
weakP    = lambda u, v, p, dx : ufl.inner(P(u, p), ufl.grad(v)) * dx
weakPres = lambda u, q, dx    : q * (J(u) - 1) * dx

'''
SOLVE PDE's
'''

V, _ = mixedSpace.sub(0).collapse()
stretch_fun = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.0))
boundaryConditions = define_bcs(V, mesh, stretch_fun)
problem = dolfinx.fem.petsc.NonlinearProblem(weakP(u, v, p, dx) + weakPres(u, q, dx), functions, boundaryConditions)
solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)

solver.rtol=1e-2
solver.atol=1e-2
solver.convergence_criterium = "incremental"

P = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
F = dolfinx.fem.FunctionSpace(mesh, P)
u1 = dolfinx.fem.Function(F)

fout = dolfinx.io.XDMFFile(mesh.comm, "myDisplacementNeoHook.xdmf", "w")
fout.write_mesh(mesh)

stretch = np.linspace(0, 0.001, 100)
for s in stretch:
    # print(f"Domain stretch: {100*s:.5f} %")
    stretch_fun.value = s   # 1, This is basically the boundary condition and is updated 
    solver.solve(functions)     # 
    u, _ = functions.split()
    u1.interpolate(u)
    fout.write_function(u1, s)