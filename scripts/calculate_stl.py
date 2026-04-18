"""
Transient heat diffusion on an STL geometry using FEniCSx (DOLFINx).

Workflow:
  STL surface mesh  →  gmsh volume mesh  →  DOLFINx mesh  →  heat solve

Same physics as calculate.py:
  dT/dt - alpha * laplace(T) = 0
  IC : T = T_init  (uniform)
  BC : -k ∂T/∂n = h(T - T_amb)  (convective cooling on all exterior faces)

Place your STL file in data/ and set STL_FILE below.
The STL must be a closed (watertight) surface.
"""

import gc
import os
import numpy as np
from mpi4py import MPI
import gmsh
import meshio

from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

# ── Parameters ────────────────────────────────────────────────────────────────
STL_FILE   = "geometries/ref_01.STL"   # path to your STL file

# material properties — glass fiber reinforced plastic (GFRP), anisotropic
# fibers run along x-axis by default; change fiber_dir to rotate
k_par      = 1.0         # conductivity along fiber direction   [W/(m·K)]
k_perp     = 0.35        # conductivity transverse to fibers    [W/(m·K)]
rho        = 1850.0      # density                              [kg/m³]
cp         = 1200.0      # specific heat                        [J/(kg·K)]
fiber_dir  = np.array([1.0, 0.0, 0.0])   # unit fiber direction vector

t_end      = 10000.0      # end time    [s]
dt         = 200.0        # time step   [s]
theta      = 1.0         # 1 = backward Euler, 0.5 = Crank-Nicolson

T_init     = 60.0       # initial temperature        [K]
T_amb      = 20.0       # ambient temperature        [K]
h_conv     = 20.0        # convective heat transfer coefficient [W/(m²·K)]
                         # ~5-25 natural convection, ~50-250 forced convection

mesh_size  = 4.0        # target element size for gmsh (in STL units)

output_file = "results/temperature_stl.xdmf"


def build_mesh_from_stl(stl_file: str, mesh_size: float):
    """
    Load a watertight STL surface and produce a tetrahedral volume mesh.
    Returns a DOLFINx mesh.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("geometry")

    # merge STL surface into gmsh
    gmsh.merge(stl_file)

    # classify surfaces and create geometry from the raw STL triangles
    gmsh.model.mesh.classifySurfaces(
        np.pi,    # angle threshold — keep all sharp edges
        True,     # boundary
        True,     # forReparametrization
        np.pi,
    )
    gmsh.model.mesh.createGeometry()

    # build a volume from the closed surface shell
    surfaces = gmsh.model.getEntities(dim=2)
    surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
    gmsh.model.geo.addVolume([surface_loop])
    gmsh.model.geo.synchronize()

    # set global mesh size and generate 3-D tet mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    # write to .msh, convert via meshio, load into DOLFINx
    gmsh.write("results/mesh.msh")
    gmsh.finalize()

    # extract only tetrahedral cells and convert to XDMF
    m = meshio.read("results/mesh.msh")
    tetra_cells = m.cells_dict["tetra"]
    meshio.write(
        "results/mesh.xdmf",
        meshio.Mesh(points=m.points, cells={"tetra": tetra_cells}),
    )

    with io.XDMFFile(MPI.COMM_WORLD, "results/mesh.xdmf", "r") as xf:
        msh = xf.read_mesh(name="Grid")
    return msh


def run():
    print(f"Loading STL: {STL_FILE}")
    msh = build_mesh_from_stl(STL_FILE, mesh_size)
    print(f"Mesh: {msh.topology.index_map(3).size_global} cells")

    V = fem.functionspace(msh, ("Lagrange", 1))

    # ── Initial condition ─────────────────────────────────────────────────────
    T_n = fem.Function(V, name="T")
    T_n.x.array[:] = T_init

    # ── Fiber direction field → written to mesh XDMF for ParaView ───────────
    n   = fiber_dir / np.linalg.norm(fiber_dir)      # ensure unit vector
    V_vec = fem.functionspace(msh, ("DG", 0, (3,)))  # vector DG0 space
    fib_f = fem.Function(V_vec, name="fiber_direction")
    fib_f.interpolate(lambda x: np.tile(n, (x.shape[1], 1)).T)

    with io.XDMFFile(msh.comm, "results/fiber_direction.xdmf", "w") as xf:
        xf.write_mesh(msh)
        xf.write_function(fib_f)

    # ── Anisotropic conductivity tensor (UFL) ────────────────────────────────
    # K = k_perp * I + (k_par - k_perp) * outer(n, n)
    K   = ufl.as_tensor(
        k_perp * np.eye(3) + (k_par - k_perp) * np.outer(n, n)
    )
    K_rcp = (1.0 / (rho * cp)) * K                   # diffusivity tensor [m²/s]

    # ── Variational form with convective Robin BC ─────────────────────────────
    # Weak form of:  rho*cp * dT/dt - div(K * grad(T)) = 0
    # with BC:       -K*grad(T)·n_out = h(T - T_amb)  on all exterior faces
    h_rcp = h_conv / (rho * cp)   # h / (rho * cp)  [m/s]

    v       = ufl.TestFunction(V)
    T_trial = ufl.TrialFunction(V)
    T_theta = theta * T_trial + (1.0 - theta) * T_n

    a = (
        T_trial * v * ufl.dx
        + dt * ufl.dot(K_rcp * ufl.grad(T_theta), ufl.grad(v)) * ufl.dx
        + dt * h_rcp * T_theta * v * ufl.ds          # convective loss
    )
    L = (
        T_n * v * ufl.dx
        + dt * h_rcp * T_amb * v * ufl.ds            # ambient heat source
    )

    problem = LinearProblem(a, L, bcs=[],
                            petsc_options={"ksp_type": "cg", "pc_type": "ilu"},
                            petsc_options_prefix="heat_stl_")

    # ── Time loop ─────────────────────────────────────────────────────────────
    with io.XDMFFile(msh.comm, output_file, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(T_n, 0.0)

        t = 0.0
        while t < t_end - 1e-12:
            t += dt
            T_n.x.array[:] = problem.solve().x.array
            xdmf.write_function(T_n, t)
            print(f"  t = {t:.1f}  T_max = {T_n.x.array.max():.2f} K"
                  f"  T_min = {T_n.x.array.min():.2f} K")

    # PETSc objects destroyed here before MPI shuts down


run()
print("Done. Results written to", output_file)
gc.collect()
os._exit(0)
