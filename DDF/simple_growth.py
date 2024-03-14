import numpy as np 
import ufl
import dolfinx as df
from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
import sys
from pathlib import Path
from enum import Enum
from dolfinx.io import XDMFFile
from petsc4py import PETSc
import basix

sys.path.insert(1, '/MyProject/MyCode/DDF/helper')

import geometry as geo
import solver as solv 
import postprocessing as pp
import simple_model

class BC(Enum):
    dirichlet = 1
    neumann = 2


#region
class bc:
    def __init__(self, bc_type, bc_condition):
        self.bc_type = bc_type
        self.bc_condition = bc_condition
    
    def info(self):
        return (self.bc_type, self.bc_condition)

'''Create Geometry'''
x_min, x_max, Nx = 0, 1, 5
y_min, y_max, Ny = 0, 1, 5
z_min, z_max, Nz = 0, 1, 5
xs = np.linspace(x_min, x_max, Nx)
ys = np.linspace(y_min, y_max, Ny)
zs = np.linspace(z_min, z_max, Nz)
X = [xs, ys, zs]
mesh = geo.create_box(X)

'''Get Functions'''
function_space, u, v = solv.get_functions(mesh)
# state_space, state, u, p, v, q = solv.get_mixed_functions(mesh)
# function_space, _, _ = solv.get_functions(mesh)
grad_u_func, grad_u_expression = solv.to_tensor_map(ufl.grad(u), mesh)

'''Create And Set Boundary Conditions'''
dirichlet_condition = lambda x : [x[0]*0, x[1]*0, x[2]*0]
no_bc, dirichlet, neumann, robin = 0, 1, 2, 3
x_left  = bc(dirichlet, lambda x : [x[0]*0, x[1]*0, x[2]*0])
x_right = bc(dirichlet, lambda x : [x[0]*1, x[1]*0, x[2]*0])
y_left  = bc(no_bc, 0)
y_right = bc(no_bc, 0)
z_left  = bc(no_bc, 0)
z_right = bc(no_bc, 0)

bc_values = [
    (x_left.info()), (x_right.info()),   # x-axis boundary type and value
    (y_left.info()), (y_right.info()),   # y-axis boundary type and value 
    (z_left.info()), (z_right.info())    # z-axis boundary type and value
    ]
#endregion

'''Create Stress Tensor'''

infinitesimal_strain = 1/2*(ufl.grad(u).T * ufl.grad(u) - ufl.Identity(len(u)))
# infinitesimal_strain = 1/2*(ufl.grad(u).T * ufl.grad(u) + ufl.grad(u).T + ufl.grad(u))
strain_function, strain_expression = solv.to_tensor_map(infinitesimal_strain, mesh)
strain_function_00, strain_expression_00 = solv.to_scalar_map(infinitesimal_strain[0, 0], mesh)
strain_function_11, strain_expression_11 = solv.to_scalar_map(infinitesimal_strain[1, 1], mesh)

'''From the paper'''
#region
'''CONSTANTS'''
# Eff_set = 0.001
Eff_set = 0.1
Ecross_set = 0.5

previous_F_gff = 1
previous_F_gcc = 1
s_l = max(strain_function_00) - Eff_set             # Equation 5
s_t = min(strain_function_00) - Ecross_set       # Equation 5

driver_function, driver_expression = simple_model.driver(u, mesh)
driver_function.interpolate(driver_expression)
F_g = simple_model.F_g(mesh, driver_function)
invF_g = ufl.inv(F_g)

I = ufl.Identity(len(u))                        # Identity tensor
F = ufl.variable(I + ufl.grad(u))               # Deformation tensor


J = ufl.det(F)                                # Determinant
F_e = ufl.variable(F*invF_g)#*pow(J, -float(1) / 3)                  # Elasticity tensor
E = 1/2*(F_e.T*F_e - I)                         # Difference tensor
# C = ufl.variable(F_e.T * F_e)                   # Ratio tensor
C = pow(J, -float(2) / 3) * F_e.T * F_e

Ic = ufl.variable(ufl.tr(C))                    # First invariant

El = df.default_scalar_type(1.0e4)
nu = df.default_scalar_type(0.3)
mu = df.fem.Constant(mesh, El / (2 * (1 + nu)))
# mu = df.fem.Constant(mesh, 1.)
lmbda = df.fem.Constant(mesh, El * nu / ((1 + nu) * (1 - 2 * nu)))

# Compressible Neo-Hookean
kappa = 1e3
# psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
psi = (mu / 2) * (Ic - 3) + kappa * (J * ufl.ln(J) - J + 1)
P = ufl.diff(psi, F)

weak_form = ufl.inner(ufl.grad(v), P) * ufl.dx(metadata={"quadrature_degree": 8})

'''Apply Boundary Conditions And Set Up Solver'''
P_with_bcs, bcs = solv.apply_boundary_conditions(mesh, bc_values, weak_form, function_space, v, X)

print("Creating problem")
problem = NonlinearProblem(P_with_bcs, u, bcs)
print("Creating solver")
solver = NewtonSolver(mesh.comm, problem)
print("Solving")
solver.solve(u)
print("Done solving")
us = []
von_Mises_stresses = []
u_new = u.copy()
us.append(u_new)
driver_function_values = []
grad_u_func_values = []
'''Solve The Problem'''
for i in np.arange(64):
    print(i)
    driver_function.interpolate(driver_expression)
    driver_function_new = driver_function.copy()
    driver_function_values.append(driver_function_new)

    grad_u_func.interpolate(grad_u_expression)
    grad_u_func_new = grad_u_func.copy()
    grad_u_func_values.append(grad_u_func_new)
    # previous_F_gff *= F_gff
    solver.solve(u)
    u_new = u.copy()
    us.append(u_new)
    print(solv.eval_expression(driver_function, mesh))

pp.write_to_paraview("simple_growth.xdmf", mesh, us)
pp.write_tensor_to_paraview("drivers.xdmf", mesh, driver_function_values)
pp.write_tensor_to_paraview("gradu.xdmf", mesh, grad_u_func_values)
# pp.write_vm_to_paraview("VonMissesStress.xdmf", mesh, von_Mises_stresses)