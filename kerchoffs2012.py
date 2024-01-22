import dolfinx.fem.petsc
import dolfinx.nls.petsc
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path


'''TODO: REMOVE THIS'''

def define_bcs(V, mesh):


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
    
    bcs = []
    for comp, bnd_fun in enumerate(bnd_funs):
        V_c, _ = V.sub(comp).collapse()          #V.sub(n) = function in n'th direction; collapse() makes it a proper (?) functionspace.
        u_fixed = dolfinx.fem.Function(V_c)                                         # recasts V_c to a dolfinx function?
        u_fixed.vector.array[:] = 0                                                 # set all values equal to zero
        dofs = dolfinx.fem.locate_dofs_geometrical(                                 # 
            (V.sub(comp), V_c), bnd_fun,                            # finds all x, y, or z values that are on boundary surface
        )                                                                           # 
        bc = dolfinx.fem.dirichletbc(u_fixed, dofs, V.sub(comp))    # mixedSpace.sub(0).sub(comp) vec -> vec -> vec
        bcs.append(bc)

    return bcs


'''CREATE MESH/FUNCTION SPACE'''

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)

P2 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)

V = dolfinx.fem.FunctionSpace(mesh, P2)

u = dolfinx.fem.Function(V)

v = ufl.TestFunction(V)

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": 4})

x = ufl.SpatialCoordinate(mesh)

def eval_expression(expr, point):
    # Determine what process owns a point and what cells it lies within
    _, _, owning_points, cells = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, point, 1e-6)
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmaps[0]
    ref_x = np.zeros((len(cells), mesh.geometry.dim),
                     dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        d_expr = dolfinx.fem.Expression(expr, ref_x, comm=MPI.COMM_SELF)
        values = d_expr.eval(mesh, np.asarray(cells).astype(np.int32))
        return values


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

s_l = dolfinx.fem.Constant(mesh, 0.0)
s_t = dolfinx.fem.Constant(mesh, 0.0)

prevF_g = F

k_ff = 1 / (1 + ufl.exp(f_lengthslope*(prevF_g[0,0] - F_ff50)))
k_cc = 1 / (1 + ufl.exp(c_thicknessslope*(prevF_g[1,1] - F_cc50)))

F_gff = ufl.conditional(ufl.ge(s_l,0),
                        k_ff*f_ffmax/(1 + ufl.exp(-f_f*(s_l - s_l50))) + 1,
                        -f_ffmax/(1 + ufl.exp(f_f*(s_l + s_l50))) + 1)

F_gcc = ufl.conditional(ufl.ge(s_t,0), 
                        ufl.sqrt(k_cc*f_ccmax/(1 + ufl.exp(-c_f*(s_t - s_t50))) + 1), 
                        ufl.sqrt(-f_ccmax/(1 + ufl.exp(c_f*(s_t - s_t50))) + 1))

F_g = ufl.as_tensor(((F_gff, 0, 0),
                     (0, F_gcc, 0),
                     (0, 0, F_gcc)))

F_e = F * ufl.inv(F_g)

E = 1/2*(F_e.T*F_e - I)    # Green-Lagrange tensor (difference tensor)

'''Strain Energy Density Function'''
Q = b_f*E[0, 0]**2 + b_t*(E[1, 1]**2 + E[2, 2]**2 + 2*E[1, 2]*E[2, 1]) + b_fr*(2*E[0, 1]*E[1, 0] + 2*E[0, 2]*E[2, 0])

E_cross = ufl.as_tensor(((E[1, 1], E[1, 2]), (E[2, 1], E[2, 2])))

'''STRAIN TENSORS'''

J = ufl.det(F_e)             # Determinant of deformation tensor
C = pow(J, -float(2 / 3)) * F_e.T * F_e       # Cauchy-Green tensor (ratio tensor)

W_pas = 1/2*C_pas*(ufl.exp(Q - 1))
W_comp = 1/2*C_comp*(J - 1)*ufl.ln(J)

W = W_pas + W_comp

P = ufl.diff(W, F)

weakP = ufl.inner(P, ufl.grad(v)) * dx 

'''SOLVE PDE's'''

boundaryConditions = define_bcs(V, mesh)

weak_form = weakP

problem = dolfinx.fem.petsc.NonlinearProblem(weak_form, u, boundaryConditions)
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



for i in range(10):

    print(eval_expression(k_ff, [0.5, 0.5, 0.5]))
    print(eval_expression(k_cc, [0.5, 0.5, 0.5]))

    solver.solve(u)

    s_l.value = i 
    s_t.value = i

    # prevF_g *= F_g

    # k_ff = 1 / (1 + ufl.exp(f_lengthslope*(prevF_g[0,0] - F_ff50)))
    # k_cc = 1 / (1 + ufl.exp(c_thicknessslope*(prevF_g[1,1] - F_cc50)))

    u1.interpolate(u)
    fout.write_function(u1, i)

fout.close()
