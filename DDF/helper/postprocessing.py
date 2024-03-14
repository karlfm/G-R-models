import dolfinx as df
import ufl
from dolfinx.io import XDMFFile
from pathlib import Path

# with XDMFFile(mesh.comm, "Hookean_facet_tags.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(facet_tags, mesh.geometry)

# "myHookeanDisplacement.xdmf"



def write_to_paraview(filename, mesh, us):

    # Not really sure about this
    AA = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
    BB = df.fem.FunctionSpace(mesh, AA)
    u1 = df.fem.Function(BB)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    fout = df.io.XDMFFile(mesh.comm, filename, "w")
    fout.write_mesh(mesh)

    for i, u in enumerate(us):
        u1.interpolate(u)
        fout.write_function(u1, i)

def write_tensor_to_paraview(filename, mesh, us):
    # Not really sure about this
    AA = ufl.TensorElement("DQ", mesh.ufl_cell(), 0)
    BB = df.fem.FunctionSpace(mesh, AA)
    u1 = df.fem.Function(BB)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    fout = df.io.XDMFFile(mesh.comm, filename, "w")
    fout.write_mesh(mesh)

    for i, u in enumerate(us):
        u1.interpolate(u)
        fout.write_function(u1, i)

def write_vm_to_paraview(filename, mesh, vms):

    # Not really sure about this
    # V_von_mises = df.fem.FunctionSpace(mesh, ("DG", 0))
    # stresses = df.fem.Function(V_von_mises)

    scalar_field = df.fem.functionspace(mesh, ("Lagrange", 1, (1, )))
    stresses = df.fem.Function(scalar_field)

    filename = Path(filename)
    filename.unlink(missing_ok=True)
    filename.with_suffix(".h5").unlink(missing_ok=True)
    foutStress = df.io.XDMFFile(mesh.comm, filename, "w")
    foutStress.write_mesh(mesh)

    for i, vm in enumerate(vms):
        print(i)
        stress_expr = df.fem.Expression(vm, scalar_field.element.interpolation_points())
        stresses.interpolate(stress_expr)
        foutStress.write_function(stresses, i)
