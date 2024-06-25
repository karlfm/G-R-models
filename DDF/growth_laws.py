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
import ddf as ddf 
import postprocessing as pp

def driver(F_e, mesh):
    f = 1/2*(F_e.T*F_e - ufl.Identity(3))
    strain_function, strain_expression = ddf.to_tensor_map(f, mesh)
    return strain_function, strain_expression

# def F_g(f1, f2):
#     return ufl.as_tensor((
#     (f1, 0, 0),
#     (0, f2, 0),
#     (0, 0, f2)))

def F_g_field(f, mesh, shape = (3,3)):
    # You have to interpolate the function like function.interpolate(expression)
    if shape is (1,1):
        tensor_field = df.fem.FunctionSpace(mesh, "DG", 0)
    else:
        tensor_field = df.fem.FunctionSpace(mesh, ufl.TensorElement("DG", mesh.ufl_cell(), 0, shape=shape))

    expression = df.fem.Expression(f, tensor_field.element.interpolation_points())
    function = df.fem.Function(tensor_field)
    return function, expression

def F_g(gamma):
    # cond = lambda a: ufl.conditional(a > 0, a, -a)
    return ufl.as_tensor((
    (pow(gamma, 2), 0, 0),
    (0, 1, 0),
    (0, 0, 1)))

def F_g1(driver_, cummulative_growth_tensor):
    # cond = lambda a: ufl.conditional(a > 0, a, -a)
    return ufl.as_tensor((
    (1 - (driver_[0, 0]), 0, 0),
    (0, 1/ufl.sqrt(1 - driver_[0, 0]), 0),
    (0, 0, 1/ufl.sqrt(1 - driver_[0, 0]))))

def F_g2(driver_):
    # cond = lambda a: ufl.conditional(a > 0, a, -a)
    return ufl.as_tensor((
    (1 + driver_[0, 0], 0, 0),
    (0, 1 + driver_[1, 1], 0),
    (0, 0, 1 + driver_[2, 2])))

def KUR(w1, w2, E_e):
    diag_elem = pow((1/2*(ufl.sqrt(2*E_e[1,1] + 1) + 1/2 - 1) + 1), 1/3)
    return ufl.as_tensor((
    (diag_elem, 0, 0),
    (0, diag_elem, 0),
    (0, 0, diag_elem)
    ))

def KFR(w1, w2, E_e):
    diag_elem = pow((1/2*(ufl.sqrt(2*E_e[1,1] + 1) + 1/2 - 1) + 1), 1/3)
    return ufl.as_tensor((
    (diag_elem, 0, 0),
    (0, diag_elem, 0),
    (0, 0, diag_elem)
    ))

def CEG(tau, F_g_f_max, F_g_f_prev, F_e_f_prev, gamma, Lambda):
    diag1 = 1/tau * pow((F_g_f_max - F_g_f_prev)/(F_g_f_max - 1), gamma) * (F_e_f_prev - Lambda) + F_g_f_prev
    diag2 = 1
    return ufl.as_tensor(((diag1, 0, 0),
                          (0, diag2, 0),
                          0, 0, diag2))



# def F_g(driver_):
#     # cond = lambda a: ufl.conditional(a > 0, a, -a)
#     return ufl.as_tensor((
#     (1 - (driver_[0, 0]), 0, 0),
#     (0, 1, 0),
#     (0, 0, 1)))

# def F_g(eff, setpoint=0.2, k=0.4):
#     return ufl.as_tensor((
#         (1 + k * (1 / (1 + ufl.exp(eff[0,0] - setpoint))), 0, 0), 
#         (0, 1, 0), 
#         (0, 0, 1))
#     )  

# def F_g(eff, setpoint=0, k=4):
#     return ufl.as_tensor(
#         ((1 + k * (1 / (1 + ufl.exp(eff - setpoint))), 0, 0), (0, 1, 0), (0, 0, 1))
#     )  

# def F_g(eff, setpoint=0.2, k=4):
#     return ufl.as_tensor(
#         ((1 + k * (1 / (1 + ufl.exp(eff - setpoint))), 0, 0), (0, 1, 0), (0, 0, 1))
#     )