import numpy as np 
import ufl
import dolfinx as df
import sys
from mpi4py import MPI
import basix
from dolfinx.io import XDMFFile


sys.path.insert(1, '/MyProject/MyCode/DDF/Geometry')

import geometry as geo

def eval_expression(expr, mesh, point = [0.5, 0.5, 0.5]):
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

def plot_boundaries(mesh, facet_tags):
    mesh.topology.create_connectivity(
        mesh.topology.dim-1, mesh.topology.dim
        )
    with XDMFFile(mesh.comm, "CheckBoundaries.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)
        
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

def dirichlet_injection(mesh, bc_values, function_space):

    all_facet_tags = {}
    boundary_conditions = []
    facet_dimension = mesh.topology.dim - 1

    points = []
    values = []
    for i, (bc_locator, component, value) in enumerate(bc_values, start=1):
        boundary_points = df.mesh.locate_entities_boundary(mesh, facet_dimension, bc_locator)
        points.append(boundary_points)
        values.append(np.ones(len(boundary_points))*i)
        boundary_dofs = df.fem.locate_dofs_topological(function_space.sub(component), facet_dimension, boundary_points)
        boundary_condition = df.fem.dirichletbc(value, boundary_dofs, function_space.sub(component))
        boundary_conditions.append(boundary_condition)

    facet_tags = df.mesh.meshtags(
        mesh, facet_dimension, np.hstack(points), np.hstack(values).astype(np.int32)
        )


    all_facet_tags[f"{component}_{value}"] = facet_tags

    return boundary_conditions, all_facet_tags

def neumann_injection(mesh, bc_info, P, v):
    n = ufl.FacetNormal(mesh)
    x = ufl.SpatialCoordinate(mesh)
    
    facet_indices, facet_markers = [], []
    fdim = mesh.topology.dim - 1
    for marker, locator, _ in bc_info:
        facets = df.mesh.locate_entities(mesh, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = df.mesh.meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

    for _, _, values in bc_info:  
        P -= ufl.inner(values * v, n) * ufl.ds(subdomain_id=marker, subdomain_data=facet_tag)

    return P

def get_functions(mesh):
    vector_element = ufl.VectorElement(family="Lagrange", cell=mesh.ufl_cell(), degree=2) #, dim=mesh.geometry.dim
    V = df.fem.FunctionSpace(mesh, vector_element)
    u = df.fem.Function(V)         # solution
    v = ufl.TestFunction(V)        # test function

    return V, u, v

def to_scalar_map(f, mesh):
    # You have to interpolate the function like function.interpolate(expression)
    scalar_field = df.fem.FunctionSpace(mesh, ("Lagrange", 1, (1, )))
    expression = df.fem.Expression(f, scalar_field.element.interpolation_points())
    function = df.fem.Function(scalar_field)
    return function, expression

def to_tensor_map(f, mesh):
    # You have to interpolate the function like function.interpolate(expression)
    tensor_field = df.fem.FunctionSpace(mesh, ufl.TensorElement("DG", mesh.ufl_cell(), 0, shape=(3,3)))
    expression = df.fem.Expression(f, tensor_field.element.interpolation_points())
    function = df.fem.Function(tensor_field)
    return function, expression

def interpolate_quadrature(ufl_expr, mesh):

    We = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree = mesh.geometry.dim, quad_scheme = "default")

    W  = df.fem.FunctionSpace(mesh, We)

    new_stress = df.fem.Function(W)

    basix_celltype = getattr(basix.CellType, mesh.topology.cell_types[0].name)
    quadrature_points, weights = basix.make_quadrature(basix_celltype,1)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = df.fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(mesh, cells)
    return np.mean(expr_eval, axis=0)

def get_measures(mesh):
    # quadrature_degree tells the computer how many and where to evaluate the points
    # "dx" tells the computer to integrate over the entire cell
    dx = ufl.Measure("dx", domain=mesh)#, metadata={"quadrature_degree": 4})
    # "ds" tells the computer to integrate over the exterior boundary
    ds = ufl.Measure("ds", domain=mesh)
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
    return

if __name__ == "__main__":
    main()

# problem = df.fem.petsc.NonlinearProblem(weak_form, u, boundaryConditions)
# solver = df.nls.petsc.NewtonSolver(mesh.comm, problem)