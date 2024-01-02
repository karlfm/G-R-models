import dolfinx.fem.petsc
import dolfinx.nls.petsc
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path


'''TODO: REMOVE THIS'''

def define_bcs(mixedSpace, mesh):


    # Gets the geometry/spatial values of the mesh and finds the smallets values in each direction
    coords = mesh.geometry.x
    xmin = min(coords[:, 0])
    ymin = min(coords[:, 1])
    zmin = min(coords[:, 2])

    # functions that will give you the closest x value of the mesh in each direction
    xmin_bnd = lambda x: np.isclose(x[0], xmin)     # Tells if point is _on_ boundary
    ymin_bnd = lambda x: np.isclose(x[1], ymin)     # Tells if point is _on_ boundary
    zmin_bnd = lambda x: np.isclose(x[2], zmin)     # Tells if point is _on_ boundary
    corner = lambda x: np.logical_and(
        np.logical_and(xmin_bnd(x), ymin_bnd(x)), zmin_bnd(x),
    )

    # A list of the functions created above
    bnd_funs = [xmin_bnd, ymin_bnd, zmin_bnd]

    # What is V0? It is the first (?) function of the mixed function space and is collapsed (?)?
    
    V0, _ = mixedSpace.sub(0).collapse()    #.sub(0) returns first subspace; .collapse() makes it a actual function space (since it is a dimension lower)

    bcs = []
    for comp, bnd_fun in enumerate(bnd_funs):
        V_c, _ = V0.sub(comp).collapse()                                            # V0.sub(0) = x etc.
        u_fixed = dolfinx.fem.Function(V_c)                                         # recasts V_c to a dolfinx function?
        u_fixed.vector.array[:] = 0                                                 # set all values equal to zero
        dofs = dolfinx.fem.locate_dofs_geometrical(                                 # 
            (mixedSpace.sub(0).sub(comp), V_c), bnd_fun,                            # finds all x, y, or z values that are on boundary surface
        )                                                                           # 
        bc = dolfinx.fem.dirichletbc(u_fixed, dofs, mixedSpace.sub(0).sub(comp))    # mixedSpace.sub(0).sub(comp) vec -> vec -> vec
        bcs.append(bc)

    return bcs


'''CREATE MESH/FUNCTION SPACE'''

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 1)

mixedSpace = dolfinx.fem.FunctionSpace(mesh, P2 * P1)

functions = dolfinx.fem.Function(mixedSpace)

u, p = ufl.split(functions)
v, q = ufl.TestFunctions(mixedSpace)

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})

'''MATH'''

# Psi_neoH = lambda u              : 2 * (ufl.tr(C(u)) - 3)  # (neo-hookian)

stretch  = lambda t              : 1 # 2 / (50**5) * t ** (5 - 1) * np.exp(-t / 50)  # from Aashilds code
sqrt_fun = lambda s              : (1 - s) ** (-0.5)

I        = lambda u              : ufl.Identity(len(u))
F_lambda = lambda u              : ufl.variable(I(u) + ufl.grad(u))

J        = lambda F              : ufl.det(F)
C        = lambda F              : pow(J(F), -float(2 / 3)) * F.T * F

F_g      = lambda s              : ufl.as_tensor(((1 - s, 0, 0), (0, sqrt_fun(s), 0), (0, 0, sqrt_fun(s))))
F_e      = lambda s, F           : ufl.variable(F * ufl.inv(F_g(s)))
e1       = lambda                : ufl.as_vector([1.0, 0.0, 0.0])

cond     = lambda a              : ufl.conditional(a > 0, a, 0)
W_hat    = lambda F              : 0.074 / (2 * 4.878) * (ufl.exp(4.878 * (ufl.tr(C(F)) - 3)) - 1)
W_f      = lambda F              : 2.628 / (2 * 5.214) * (ufl.exp(5.214 * cond(ufl.inner(C(F) * e1(), e1()) - 1) ** 2) - 1)
Psi_Holz = lambda F              : W_hat(F) + W_f(F)                                            
P        = lambda s, p, F        : ufl.diff(Psi_Holz(F_e(s, F)), F) + p * J(F_e(s, F)) * ufl.inv(F_e(s, F).T)   # Referemce Tensor
weakP    = lambda s, F, v, p, dx : ufl.inner(P(s, p, F_e(s, F)), ufl.grad(v)) * dx                              # Elasticity term
weakPres = lambda F, q, dx       : q * (J(F) - 1) * dx                                                          # Pressure term??

'''SOLVE PDE's'''

time1 = np.linspace(0, 0.5, 500)
time2 = 0.5*np.ones(500)
time  = np.concatenate((time1, time2))

active_fun = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))

boundaryConditions = define_bcs(mixedSpace, mesh)

F = F_lambda(u)

weak_form = weakP(active_fun, F, v, p, dx) + weakPres(F, q, dx)

problem = dolfinx.fem.petsc.NonlinearProblem(weak_form, functions, boundaryConditions)
solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)

solver.rtol = 1e-4
solver.atol = 1e-4
solver.convergence_criterium = "incremental"

# Not really sure about this
AA = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
BB = dolfinx.fem.FunctionSpace(mesh, AA)
u1 = dolfinx.fem.Function(BB)

filename = Path("myDisplacementHolz.xdmf")
filename.unlink(missing_ok=True)
filename.with_suffix(".h5").unlink(missing_ok=True)
fout = dolfinx.io.XDMFFile(mesh.comm, filename, "w")
fout.write_mesh(mesh)

for (i, a) in enumerate(time):
    if (i+1) % 100 == 0:
        print(f"Growth tensor: {str(F_g(a))}; step {i+1}")
        print(f"Elasticity tensor: {str(F_e(a, F))}; step {i+1}")
    active_fun.value = a #bytt med iopodarterubg op F_g
    # F_g(adisa) = ny verdi
    solver.solve(functions)
    u, _ = functions.split()
    u1.interpolate(u)
    fout.write_function(u1, i)

fout.close()
