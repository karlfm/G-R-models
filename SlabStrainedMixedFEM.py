import dolfinx.fem.petsc
import dolfinx.nls.petsc
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc


'''TODO: REMOVE THIS'''

def define_bcs(mixedSpace, mesh):
    """
    Defines boundary conditions based on displacement, assuming the domain
    has a box-like shape. We'll keep the displacement on the sides defined
    by lowest x coord, y coords and z coords fixed in their respective planes.

    Args:
        state_space (FunctionSpace): function space for displacement and pressure
        mesh (dolfinx.Mesh): Domain in which we solve the problem

    Returns:
        List of boundary conditions
    """

    coords = mesh.geometry.x
    xmin = min(coords[:, 0])
    ymin = min(coords[:, 1])
    zmin = min(coords[:, 2])

    xmin_bnd = lambda x: np.isclose(x[0], xmin)
    ymin_bnd = lambda x: np.isclose(x[1], ymin)
    zmin_bnd = lambda x: np.isclose(x[2], zmin)
    corner = lambda x: np.logical_and(
        np.logical_and(xmin_bnd(x), ymin_bnd(x)), zmin_bnd(x),
    )

    bcs = []

    # fix three of the boundaries in their respective planes

    bnd_funs = [xmin_bnd, ymin_bnd, zmin_bnd]
    components = [0, 1, 2]

    V0, _ = mixedSpace.sub(0).collapse()

    for bnd_fun, comp in zip(bnd_funs, components):
        V_c, _ = V0.sub(comp).collapse()
        u_fixed = dolfinx.fem.Function(V_c)
        u_fixed.vector.array[:] = 0
        dofs = dolfinx.fem.locate_dofs_geometrical(
            (mixedSpace.sub(0).sub(comp), V_c), bnd_fun,
        )
        bc = dolfinx.fem.dirichletbc(u_fixed, dofs, mixedSpace.sub(0).sub(comp))
        bcs.append(bc)

    # fix corner completely

    u_fixed = dolfinx.fem.Function(V0)
    u_fixed.vector.array[:] = 0
    dofs = dolfinx.fem.locate_dofs_geometrical((mixedSpace.sub(0), V0), corner)
    bc = dolfinx.fem.dirichletbc(u_fixed, dofs, mixedSpace.sub(0))
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

act_strn = lambda t              : 2 / (50**5) * t ** (5 - 1) * np.exp(-t / 50)  # from Aashilds code
sqrt_fun = lambda s              : (1 - s) ** (-0.5)
I        = lambda u              : ufl.Identity(len(u))
F        = lambda u              : ufl.variable(I(u) + ufl.grad(u))
F_a      = lambda s              : ufl.as_tensor(((1 - s, 0, 0), (0, sqrt_fun(s), 0), (0, 0, sqrt_fun(s))))
F_e      = lambda s, u           : ufl.variable(F(u) * ufl.inv(F_a(s)))
J        = lambda u              : ufl.det(F(u))
C        = lambda u              : pow(J(u), -float(2 / 3)) * F(u).T * F(u)
e1       = lambda                : ufl.as_vector([1.0, 0.0, 0.0])
cond     = lambda a              : ufl.conditional(a > 0, a, 0)
Psi_neoH = lambda u              : 2 * (ufl.tr(C(u)) - 3)  # (neo-hookian)
W_hat    = lambda u              : 0.074 / (2 * 4.878) * (ufl.exp(4.878 * (ufl.tr(C(u)) - 3)) - 1)
W_f      = lambda u              : 2.628 / (2 * 5.214) * (ufl.exp(5.214 * cond(ufl.inner(C(u) * e1(), e1()) - 1) ** 2) - 1)
Psi_Holz = lambda u              : W_hat(u) + W_f(u)
P        = lambda s, p, u        : ufl.det(F_a(s)) * ufl.diff(Psi_Holz(u), F_e(s, u)) * ufl.inv(F_a(s).T) + p * J(u) * ufl.inv(F(u).T)
weakP    = lambda s, u, v, p, dx : ufl.inner(P(s, p, u), ufl.grad(v)) * dx  # elasticity term
weakPres = lambda u, q, dx       : q * (J(u) - 1) * dx  # pressure term??

'''SOLVE PDE's'''

time = np.linspace(0, 1000, 1000)
active_values = act_strn(time)
active_fun = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0))

boundaryConditions = define_bcs(mixedSpace, mesh)

weak_form = weakP(active_fun, u, v, p, dx) + weakPres(u, q, dx)

problem = dolfinx.fem.petsc.NonlinearProblem(weak_form, functions, boundaryConditions)
solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)

solver.rtol = 1e-4
solver.atol = 1e-4
solver.convergence_criterium = "incremental"

AA = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
BB = dolfinx.fem.FunctionSpace(mesh, AA)
u1 = dolfinx.fem.Function(BB)

fout = dolfinx.io.XDMFFile(mesh.comm, "myDisplacementHolz.xdmf", "w")
fout.write_mesh(mesh)

for (i, a) in enumerate(active_values):
    if i % 50 == 0:
        print(f"Active tension value: {a:.2f}; step {i}")
    active_fun.value = a
    solver.solve(functions)
    u, _ = functions.split()
    u1.interpolate(u)
    fout.write_function(u1, a)
fout.close
