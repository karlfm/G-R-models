import numpy as np 
import ufl
import dolfinx as df
import sys
from mpi4py import MPI


sys.path.insert(1, '/MyProject/MyCode/DDF/Geometry')

import geometry as geo

def eval_expression(expr, point, mesh):
    # Determine what process owns a point and what cells it lies within
    _, _, owning_points, cells = df.cpp.geometry.determine_point_ownership(
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
        d_expr = df.fem.Expression(expr, ref_x, comm=MPI.COMM_SELF)
        values = d_expr.eval(mesh, np.asarray(cells).astype(np.int32))
        return values

def set_boundary_types(mesh, boundary_type, X):
    # boundary_conditions is a list of pairs of ints containing boundary conditions for each dimension
    boundary_conditions_dolfin = []
    flattened_boundaries = []
    boundaries = geo.get_boundary_nodes(mesh, X)

    dimension = mesh.topology.dim
    boundary_dimension = dimension - 1

    for boundary_nodes, boundary_condition in zip(boundaries, boundary_type):
        x_min = np.full(len(boundary_nodes[0]), boundary_condition[0]).astype(np.int32)
        x_max = np.full(len(boundary_nodes[1]), boundary_condition[1]).astype(np.int32)
        boundary_conditions_dolfin.extend([x_min, x_max])
        flattened_boundaries.extend(boundary_nodes)

    flattened_boundaries = np.hstack(flattened_boundaries)
    boundary_conditions_dolfin = np.hstack(boundary_conditions_dolfin)
    sort_array = np.argsort(flattened_boundaries)
    facet_tags = df.mesh.meshtags(
        mesh, boundary_dimension, flattened_boundaries[sort_array], boundary_conditions_dolfin[sort_array]
    )


    return facet_tags

def apply_boundary_conditions(mesh, facet_tags, values, P, function_space, v):
    # v is a test function
    # values is a list of pairs (type, lambda) that take in coordinates and return values?
    dx, ds, dS = get_measures(mesh, facet_tags)
    no_bc, dirichlet, neumann, robin = 0, 1, 2, 3

    boundary_conditions = []
    dimension = mesh.topology.dim
    boundary_dimension = dimension - 1
    for bc_type, func in values:
        # for dirichelt bcs
        if bc_type == dirichlet:
            boundary_u = df.fem.Function(function_space)
            boundary_u.interpolate(func)
            facets = facet_tags.find(bc_type)
            dofs = df.fem.locate_dofs_topological(function_space, boundary_dimension, facets)
            dirichlet_boundary_condition = df.fem.dirichletbc(boundary_u, dofs)
            boundary_conditions.append(dirichlet_boundary_condition)
        elif bc_type == neumann:
            # P += ufl.dot(func, v) *  ds(bc_type)
            P -= ufl.inner(func, v) *  ds(bc_type)

    return P, boundary_conditions
    # for neumann bcs

def get_functions(mesh):
    #Q: WHY DOES THIS ONLY WORK WHEN LAGRANGE = 1??
    V = df.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
    u = df.fem.Function(V)         # solution
    v = ufl.TestFunction(V)        # test function

    return V, u, v


def get_measures(mesh, facet_tags):
    # quadrature_degree tells the computer how many and where to evaluate the points
    # 
    # "dx" tells the computer to integrate over the entire cell
    dx = ufl.Measure("dx", domain=mesh)#, metadata={"quadrature_degree": 4})
    # "ds" tells the computer to integrate over the exterior boundary
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
                     #, metadata={"quadrature_degree": 4})
    # "dS" tells the computer to integrate over the interior boundary
    dS = ufl.Measure("dS", domain=mesh)#, metadata={"quadrature_degree": 4})
    return dx, ds, dS

def get_basic_tensors(u):

    I = ufl.Identity(len(u))                      # Identity tensor
    F = ufl.variable(I + ufl.grad(u))             # Deformation tensor
    E = 1/2*(F.T*F - I)                           # Difference tensor
    J = ufl.det(F)                                # Determinant
    # C = pow(J, -float(2 / 3)) * F.T * F           # Ratio tensor
    C = ufl.variable(F.T * F)
    return I, F, J, C, E

def get_stress_tensor(Psi, F):
    sigma = ufl.diff(Psi, F)
    return sigma

def weak_form(sigma, test_function, dx):
    ufl.inner(sigma, ufl.grad(test_function)) * dx

def rhs(v, Psi, F, T, dx, ds):
    # v = test function
    # Psi = strain energy function
    # F = deformation tensor
    # T = boundary condition
    # dx = I think a quadrature version of normal dx
    # ds = I think a quadrature version of normal dx but 1D less
    
    P = ufl.diff(Psi, F)    # I think first Piola-Kirchoff tensor?
    F_new = ufl.inner(P, ufl.grad(v)) * dx - ufl.inner(T, v)*ds     # Not sure about BC

    return F_new

def solver(F, u, bcs):
    # F is deformation tensor
    # u is solution
    # bcs is a list of boundary condiiotns
    problem = df.fem.petsc.NonlinearProblem(F, u, bcs)
    return problem


def main():
    dx, ds, dS = get_measures()

if __name__ == "__main__":
    main()

# problem = df.fem.petsc.NonlinearProblem(weak_form, u, boundaryConditions)
# solver = df.nls.petsc.NewtonSolver(mesh.comm, problem)