import dolfinx
import ufl
from dolfinx.io import XDMFFile
from pathlib import Path
import basix

def write_scalar_to_paraview(filename, mesh, scalars):

    function_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))#, (1,  )
    function = dolfinx.fem.Function(function_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    foutStress = dolfinx.io.XDMFFile(mesh.comm, filename, "w")
    foutStress.write_mesh(mesh)

    for i, scalar in enumerate(scalars):
        stress_expr = dolfinx.fem.Expression(scalar, function_space.element.interpolation_points())
        function.interpolate(stress_expr)
        foutStress.write_function(function, i)
    
def write_vector_to_paraview(filename, mesh, us):

    vector_element = basix.ufl.element(family="Lagrange", cell=str(mesh.ufl_cell()), degree=1, shape=(3,))
    function_space = dolfinx.fem.functionspace(mesh, vector_element)
    function = dolfinx.fem.Function(function_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    fout = dolfinx.io.XDMFFile(mesh.comm, filename, "w")
    fout.write_mesh(mesh)

    for i, u in enumerate(us):
        function.interpolate(u)
        fout.write_function(function, i)

def write_tensor_to_paraview(filename, mesh, tensors):

    tensor_element = ufl.TensorElement("DG", mesh.ufl_cell(), 0)
    function_space = dolfinx.fem.FunctionSpace(mesh, tensor_element)
    function = dolfinx.fem.Function(function_space)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    fout = dolfinx.io.XDMFFile(mesh.comm, filename, "w")
    fout.write_mesh(mesh)
   
    for i, tensor in enumerate(tensors):
        function.interpolate(tensor)
        fout.write_function(function, i)
