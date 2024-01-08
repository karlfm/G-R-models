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

'''CONSTANTS'''

f_ffmax = 0.3
f_f = 150
s_l50 = 0.06
F_ff50 = 1.35
f_lengthslope = 40
f_ccmax = 0.1
c_f = 75
s_t50 = 0.07
F_cc50 = 1.28
c_thicknessslope = 60

'''MATH'''

k_ff     = lambda s              : 1 / (1 + exp_fun(f_lengthslope*(s - F_ff50)))
kerchoff = lambda s              : k_ff(s)#*(f_ffmax/1 + np.exp(-f_f(1 - s_l50)))
F_giff   = lambda s              : ufl.as_tensor(((kerchoff(s), 0, 0), (0, sqrt_fun(kerchoff(s)), 0), (0, 0, sqrt_fun(kerchoff(s)))))

stretch  = lambda t              : 1 # 2 / (50**5) * t ** (5 - 1) * np.exp(-t / 50)  # from Aashilds code
sqrt_fun = lambda s              : (1 - s) ** (-0.5)
exp_fun  = lambda s              : (1 + 0.5*s + 1/9*s**2 + 1/72*s**3 + 1/1008*s**4 + 1/30240*s**5)/(1 - 0.5*s + 1/9*s**2 - 1/72*s**3 + 1/1008*s**4 - 1/30240*s**5)
sqrt_fun = lambda s              : (1 + 5/4*s + 5/16*s**2)/(1 + 3/4*s + 1/16*s**2)

I        = lambda u              : ufl.Identity(len(u))
F_lambda = lambda u              : ufl.variable(I(u) + ufl.grad(u))

J        = lambda F              : ufl.det(F)
C        = lambda F              : pow(J(F), -float(2 / 3)) * F.T * F
# E        = lambda F              : 1/2*(F.T*F - I(F))

F_g      = lambda s              : ufl.as_tensor(((1 - s, 0, 0), (0, exp_fun(s), 0), (0, 0, exp_fun(s))))
F_e      = lambda s, F           : ufl.variable(F * ufl.inv(F_giff(s)))

neoHook  = lambda F              : ufl.tr(C(F))
P        = lambda s, p, F        : (ufl.diff(neoHook(F_e(s, F)), F) + p * J(F_e(s, F)) * ufl.inv(F_e(s, F).T))   # Referemce Tensor
weakP    = lambda s, F, v, p, dx : ufl.inner(P(s, p, F_e(s, F)), ufl.grad(v)) * dx                              # Elasticity term
weakPres = lambda F, q, dx       : q * (J(F) - 1) * dx                                                          # Pressure term??

'''SOLVE PDE's'''

time1 = np.linspace(0, 1, 500)
time2 = np.ones(500)
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

filename = Path("myDisplacement.xdmf")
filename.unlink(missing_ok=True)
filename.with_suffix(".h5").unlink(missing_ok=True)
fout = dolfinx.io.XDMFFile(mesh.comm, filename, "w")
fout.write_mesh(mesh)

for (i, a) in enumerate(time):
    if (i+1) % 100 == 0:
        print(f"Growth tensor: {str(F_g(a))}; step {i+1}")
        print(f"Elasticity tensor: {str(F_e(a, F))}; step {i+1}")
    active_fun.value = a
    solver.solve(functions)
    u, _ = functions.split()
    u1.interpolate(u)
    fout.write_function(u1, i)

fout.close()
