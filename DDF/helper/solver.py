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

def apply_boundary_conditions(mesh, bc_values, P, function_space, v, X):
    boundary_type = [(bc_values[i][0], bc_values[i + 1][0]) for i in range(0, len(bc_values), 2)]
    facet_tags = set_boundary_types(mesh, boundary_type, X)
    # plot_boundaries(mesh, facet_tags)
    dx, ds, dS = get_measures(mesh)
    no_bc, dirichlet, neumann, robin = 0, 1, 2, 3

    boundary_conditions = []
    dimension = mesh.topology.dim
    boundary_dimension = dimension - 1
    # uncomment line below if you are in a 2D mixed function space
    # function_space = function_space.sub(0).collapse() 
    coords = mesh.geometry.x

    for value in bc_values:
        bc_type, func = value
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


def apply_boundary_conditions_with_pressure(mesh, values, P, state_space, v, X):
    # v is a test function
    # values is a list of pairs (type, lambda) that take in coordinates and return values
    # boundary_type = [(values[i][0], values[i + 1][0]) for i in range(0, len(values), 2)]
    # facet_tags = set_boundary_types(mesh, boundary_type, X)
    # dx, ds, dS = get_measures(mesh, facet_tags)
    # no_bc, dirichlet, neumann, robin = 0, 1, 2, 3
    # V, _ = state_space.sub(0).collapse()

    # boundary_conditions = []
    # dimension = mesh.topology.dim
    # boundary_dimension = dimension - 1

    # num_components = state_space.num_sub_spaces
    # V0, _ = state_space.sub(0).collapse()    

    # for value in values:
    #     bc_type, func = value
    #     if bc_type == dirichlet:
    #         boundary_u = df.fem.Function(V)
    #         boundary_u.interpolate(func)  # Use the i'th component of the boundary function
    #         breakpoint()
    #         facets = facet_tags.find(bc_type)
    #         dofs = df.fem.locate_dofs_topological(V, boundary_dimension, facets)
    #         dirichlet_boundary_condition = df.fem.dirichletbc(boundary_u, dofs)#, state_space.sub(0).sub(component)
    #         boundary_conditions.append(dirichlet_boundary_condition)

    #     elif bc_type == neumann:
    #         P -= ufl.inner(func, v) * ds(bc_type)  # Use the i'th component of the test and boundary functions
    # bcs = []
    # boundary_type = [(values[i][0], values[i + 1][0]) for i in range(0, len(values), 2)]
    # facet_tags = set_boundary_types(mesh, boundary_type, X)

    # return P, bcs

    coords = mesh.geometry.x
    xmin = min(coords[:, 0])
    ymin = min(coords[:, 1])
    zmin = min(coords[:, 2])

    xmin_bnd = lambda x : np.isclose(x[0], xmin)
    ymin_bnd = lambda x : np.isclose(x[1], ymin)
    zmin_bnd = lambda x : np.isclose(x[2], zmin)

    bcs = []
    
    # fix three of the boundaries in their respective planes
    
    bnd_funs = [xmin_bnd, ymin_bnd, zmin_bnd]
    components = [0, 1, 2]
    
    V0, _ = state_space.sub(0).collapse()    

    for bnd_fun, comp in zip(bnd_funs, components):
        V_c, _ = V0.sub(comp).collapse()
        u_fixed = df.fem.Function(V_c)
        u_fixed.vector.array[:] = 0
        dofs = df.fem.locate_dofs_geometrical((state_space.sub(0).sub(comp),V_c), bnd_fun)
        bc = df.fem.dirichletbc(u_fixed, dofs, state_space.sub(0).sub(comp))
        bcs.append(bc)
    
    return P, bcs


def get_functions(mesh):
    vector_element = ufl.VectorElement(family="Lagrange", cell=mesh.ufl_cell(), degree=1) #, dim=mesh.geometry.dim
    V = df.fem.FunctionSpace(mesh, vector_element)
    u = df.fem.Function(V)         # solution
    v = ufl.TestFunction(V)        # test function

    return V, u, v

def get_mixed_functions(mesh):
    #DOESNT WORK FOR ANY DEGREE OTHER THAN TWO AN ELEMENTS THAT ARENT TRIANGULAR
    V = ufl.VectorElement(family="Lagrange", cell=mesh.ufl_cell(), degree=1) #, dim=mesh.geometry.dim
    P = ufl.FiniteElement(family="Lagrange", cell=mesh.ufl_cell(), degree=1)
    
    state_space = df.fem.FunctionSpace(mesh, V * P)
    state = df.fem.Function(state_space)

    u, p = ufl.split(state)
    v, q = ufl.TestFunctions(state_space)
    
    return state_space, state, u, p, v, q

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