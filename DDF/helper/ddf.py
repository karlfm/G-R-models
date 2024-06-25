import numpy as np 
import ufl
import dolfinx
import sys
from mpi4py import MPI
import basix
from dataclasses import dataclass
from dolfinx.io import XDMFFile
from typing import Protocol

sys.path.insert(1, '/MyProject/MyCode/DDF/Geometry')
import geometry as geo

def eval_expression(expr, mesh, point = [0.5, 0.5, 0.5]):
    # Determine what process owns a point and what cells it lies within
    arg1=np.array(point, dtype=np.float64)
    _, _, owning_points, cells  = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, arg1 , 1e-6)
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmap
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
    facet_tags = dolfinx.mesh.meshtags(
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
        boundary_points = dolfinx.mesh.locate_entities_boundary(mesh, facet_dimension, bc_locator)
        points.append(boundary_points)
        values.append(np.ones(len(boundary_points))*i)  # This is just for the facet tags; not the boundary condition
        boundary_dofs = dolfinx.fem.locate_dofs_topological(function_space.sub(component), facet_dimension, boundary_points)
        boundary_condition = dolfinx.fem.dirichletbc(value, boundary_dofs, function_space.sub(component))
        boundary_conditions.append(boundary_condition)

    facet_tags = dolfinx.mesh.meshtags(
        mesh, facet_dimension, np.hstack(points), np.hstack(values).astype(np.int32)
        )

    all_facet_tags[f"{component}_{value}"] = facet_tags

    return boundary_conditions, all_facet_tags

def dirichlet_injection_ellipsoid(geometry, bc_values, function_space):

    boundary_conditions = []
    facet_dimension = geometry.mesh.topology.dim - 1


    for bc_location, component, value in bc_values:
        boundary_points = geometry.ffun.find(geometry.markers[bc_location][0])  # Specify the marker used on the boundary
        geometry.mesh.topology.create_connectivity(
            geometry.mesh.topology.dim - 1,
            geometry.mesh.topology.dim,
        )
        boundary_dofs = dolfinx.fem.locate_dofs_topological(function_space.sub(component), facet_dimension, boundary_points)
        boundary_condition = dolfinx.fem.dirichletbc(value, boundary_dofs, function_space.sub(component))
        boundary_conditions.append(boundary_condition)

    return boundary_conditions

def neumann_injection_ellipsoid(geometry, bc_info, P, F, v):

    neumann_bcs = []

    n = ufl.FacetNormal(geometry.mesh)

    for bc_location, value in bc_info:   # TODO: implement shear Neumann BC
        # n = value * ufl.cofac(F) * N
        neumann_bcs.append(ufl.inner(v, value * ufl.cofac(F) * n) * ufl.ds(subdomain_id=geometry.markers[bc_location][0], subdomain_data=geometry.ffun))

    return neumann_bcs

def robin_injection_ellipsoid(geometry, bc_info, P, F, v, u):

    robin_bcs = []

    n = ufl.FacetNormal(geometry.mesh)

    for bc_location, w1, w2 in bc_info:   # TODO: implement shear Neumann BC
        # n = value * ufl.cofac(F) * N
        robin_bcs.append(w1*(ufl.inner(v, (u - w2 * ufl.cofac(F) * n))) * ufl.ds(subdomain_id=geometry.markers[bc_location][0], subdomain_data=geometry.ffun))

    return robin_bcs

def neumann_injection(mesh, bc_info, P, v):
    n = ufl.FacetNormal(mesh)
    
    facet_indices, facet_markers = [], []
    facet_dimension = mesh.topology.dim - 1
    for marker, locator, _ in bc_info:
        facets = dolfinx.mesh.locate_entities(mesh, facet_dimension, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dolfinx.mesh.meshtags(mesh, facet_dimension, facet_indices[sorted_facets], facet_markers[sorted_facets])

    for marker, _, values in bc_info:  
        P -= ufl.inner(values * v, n) * ufl.ds(subdomain_id=marker, subdomain_data=facet_tag)

    return P






def get_facet_tags(mesh, marker, locator):
        
    facet_indices, facet_markers = [], []
    facet_dimension = mesh.topology.dim - 1
    facets = dolfinx.mesh.locate_entities(mesh, facet_dimension, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = dolfinx.mesh.meshtags(mesh, facet_dimension, facet_indices[sorted_facets], facet_markers[sorted_facets])

    return facet_tag

def get_functions(mesh):
    # vector_element = ufl.vectorElement(family="Lagrange", cell=mesh.ufl_cell(), degree=1) #, dim=mesh.geometry.dim
    vector_element = basix.ufl.element(family="Lagrange", cell=str(mesh.ufl_cell()), degree=1, shape=(mesh.geometry.dim, ))
    V = dolfinx.fem.functionspace(mesh, vector_element)
    u = dolfinx.fem.Function(V)         # solution
    v = ufl.TestFunction(V)        # test function

    return V, u, v

def to_scalar_map(f, mesh):
    # You have to interpolate the function like function.interpolate(expression)
    scalar_field = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (1, )))
    expression = dolfinx.fem.Expression(f, scalar_field.element.interpolation_points())
    function = dolfinx.fem.Function(scalar_field)
    return function, expression

def to_tensor_map(f, mesh, shape = (3,3)):
    # You have to interpolate the function like function.interpolate(expression)
    if shape is (1,1):
        tensor_field = dolfinx.fem.functionspace(mesh, "DG", 0)
    else:
        tensor_field = dolfinx.fem.functionspace(mesh, basix.ufl.element(family="DG", cell=str(mesh.ufl_cell()), degree=0, shape=shape))

    expression = dolfinx.fem.Expression(f, tensor_field.element.interpolation_points())
    function = dolfinx.fem.Function(tensor_field)
    
    return function, expression

def interpolate_quadrature(ufl_expr, mesh):

    We = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree = mesh.geometry.dim, quad_scheme = "default")

    W  = dolfinx.fem.functionspace(mesh, We)

    new_stress = dolfinx.fem.Function(W)

    basix_celltype = getattr(basix.CellType, mesh.topology.cell_types[0].name)
    quadrature_points, weights = basix.make_quadrature(basix_celltype,1)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    expr_expr = dolfinx.fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(mesh, cells)
    return np.mean(expr_eval, axis=0)

def main():
    return

if __name__ == "__main__":
    main()
