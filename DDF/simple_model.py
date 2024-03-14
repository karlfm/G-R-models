import numpy as np
import dolfinx as df
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from pathlib import Path
import copy
import sys

from dolfinx.io import XDMFFile

sys.path.insert(1, '/MyProject/MyCode/DDF/helper')

import geometry as geo
import solver as solv 
import postprocessing as pp

def driver(u, mesh):
    f = 1/2*(ufl.grad(u).T * ufl.grad(u) - ufl.Identity(len(u)))
    strain_function, strain_expression = solv.to_tensor_map(f, mesh)
    return strain_function, strain_expression

def F_g(mesh, driver_):
    # cond = lambda a: ufl.conditional(a > 0, a, -a)
    return ufl.as_tensor((
    (1, 0, 0), #+ (driver_[0, 0]),
    (0, 1, 0),
    (0, 0, 1)))

