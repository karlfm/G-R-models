import dolfinx as df
import ufl
from dolfinx.io import XDMFFile
from pathlib import Path

def write_scalar_to_paraview(filename, mesh, scalars):

    function_space = df.fem.functionspace(mesh, ("Lagrange", 1))#, (1,  )
    function = df.fem.Function(function_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    foutStress = df.io.XDMFFile(mesh.comm, filename, "w")
    foutStress.write_mesh(mesh)

    for i, scalar in enumerate(scalars):
        stress_expr = df.fem.Expression(scalar, function_space.element.interpolation_points())
        function.interpolate(stress_expr)
        foutStress.write_function(function, i)
    
def write_vector_to_paraview(filename, mesh, us):

    vector_element = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    function_space = df.fem.FunctionSpace(mesh, vector_element)
    function = df.fem.Function(function_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    fout = df.io.XDMFFile(mesh.comm, filename, "w")
    fout.write_mesh(mesh)

    for i, u in enumerate(us):
        function.interpolate(u)
        fout.write_function(function, i)

def write_tensor_to_paraview(filename, mesh, tensors):

    tensor_element = ufl.TensorElement("DG", mesh.ufl_cell(), 0)
    function_space = df.fem.FunctionSpace(mesh, tensor_element)
    function = df.fem.Function(function_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    fout = df.io.XDMFFile(mesh.comm, filename, "w")
    fout.write_mesh(mesh)
   
    for i, tensor in enumerate(tensors):
        function.interpolate(tensor)
        fout.write_function(function, i)
