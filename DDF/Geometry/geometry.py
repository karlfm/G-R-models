import dolfinx as df 
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx.io import XDMFFile
import sys

sys.path.insert(1, '/MyProject/MyCode/DDF/SpatialSolver')

import solver as solv 

# QUESTIONS
# 1, HOW IS VERTEX NUMBERING DEFINED?
# 2, HOW DO YOU DECIDE WHAT TYPES OF FORMS YOU WANT? 
# 3, WHAT DETER MINES HOW MANY NODES YOU HAVE. 
### 3 a, WHY DO I HAVE 18 OR SO NODES IN  A 1x1x1 CUBE? I WOULD'VE EXPECTED 8

def create_unit_cube(xs, ys, zs):
    return df.mesh.create_unit_cube(MPI.COMM_WORLD, xs, ys, zs)     # IDK what the first arg is

def create_unit_square(xs, ys):
    return df.mesh.create_unit_square(MPI.COMM_WORLD, xs, ys)     # IDK what the first arg is

def create_box(xs, ys, zs):
    return df.mesh.create_box(MPI.COMM_WORLD, [np.array([xs[0], ys[0], zs[0]]), np.array([xs[-1], ys[-1], zs[-1]])], [len(xs), len(ys), len(zs)], cell_type=df.mesh.CellType.hexahedron)

def get_boundary_nodes(mesh):
    # mesh is a dolfinx.mesh.create_mesh_... thing.
    output = []
    dimension = mesh.topology.dim
    boundary_dimension = dimension - 1

    for i in range(dimension):
        x_min_edge = mesh.geometry.x[:, i][0]
        x_max_edge = mesh.geometry.x[:, i][-1]

        x_min = df.mesh.locate_entities(mesh, boundary_dimension, lambda x: np.isclose(x[i], x_min_edge))
        x_max = df.mesh.locate_entities(mesh, boundary_dimension, lambda x: np.isclose(x[i], x_max_edge))
        output.append((x_min, x_max))

    return output

def main():

    mesh = unit_cube(1, 1, 1)
    boundary_conditions = [(1,1), (2,2), (3,3)]
    facet_tags = solve.set_boundary_types(mesh, boundary_conditions)
    
    ### WHAT DOES THIS DO?
    mesh.topology.create_connectivity(
        mesh.topology.dim-1, mesh.topology.dim
        )
    
    with XDMFFile(mesh.comm, "facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)

    # boundary_nodes = get_boundary_nodes(mesh)
    # print(boundary_nodes)
    breakpoint()

if __name__ == "__main__":
    main()