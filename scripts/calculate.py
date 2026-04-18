"""
Transient heat diffusion using FEniCSx (DOLFINx).

Models a hot 3-D block cooling down in ambient air:
  - Initial condition : uniform T_init throughout the block
  - Boundary condition: T = T_amb on all faces (block surface meets ambient air)
  - Solves: dT/dt - alpha * laplace(T) = 0

Time discretisation: backward Euler (theta = 1).
Output: XDMF (open with ParaView).
"""

import gc
import os
import numpy as np
from mpi4py import MPI

from dolfinx import fem, io, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
import ufl

from thermosim.pore_generator import PoreGenerator

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny, nz = 16, 16, 16  # mesh resolution (keep lower in 3-D to limit DOFs)
# material properties (default: steel)
k       = 50.0       # thermal conductivity  [W/(m·K)]
rho     = 7800.0     # density               [kg/m³]
cp      = 500.0      # specific heat         [J/(kg·K)]
alpha   = k / (rho * cp)  # thermal diffusivity   [m²/s]
t_end   = 3000.0       # end time             [s]
dt      = 10.0      # time step            [s]
theta   = 1.0        # 1 = backward Euler, 0.5 = Crank-Nicolson

T_init  = 500.0      # initial plate temperature  [K]
T_amb   = 293.0      # ambient temperature         [K]

# pore / inclusion settings
use_pores  = True    # set False to run without pores (pure homogeneous block)
n_pores    = 1      # number of pores to place
r_min      = 0.2    # minimum pore radius  [m, in unit-cube coords]
r_max      = 0.3    # maximum pore radius  [m]
alpha_pore = .5e-12  # diffusivity inside pores (air-like)  [m²/s]
pore_seed  = 42      # random seed for reproducibility

output_file = "results/temperature.xdmf"

def run():
    # ── Mesh & function space ─────────────────────────────────────────────────
    msh = dmesh.create_unit_cube(MPI.COMM_WORLD, nx, ny, nz)
    V   = fem.functionspace(msh, ("Lagrange", 1))

    # ── Diffusivity field (uniform or porous) ────────────────────────────────
    if use_pores:
        pores = PoreGenerator(n_pores=n_pores, r_min=r_min, r_max=r_max, seed=pore_seed)
        pores.summary()
        # DG0: one value per cell at centroid — sharp pore boundaries
        alpha_field = pores.make_diffusivity_field(msh, alpha, alpha_pore)
    else:
        V_dg = fem.functionspace(msh, ("DG", 0))
        alpha_field = fem.Function(V_dg, name="alpha")
        alpha_field.x.array[:] = alpha
        print("Pores disabled — homogeneous block.")

    # ── Boundary condition  (T = T_amb on all six faces) ─────────────────────
    boundary_dofs = fem.locate_dofs_geometrical(V, lambda x: (
        np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) |
        np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0) |
        np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)
    ))
    bc = fem.dirichletbc(T_amb, boundary_dofs, V)

    # ── Initial condition  (uniform T_init everywhere) ───────────────────────
    T_n = fem.Function(V, name="T")
    T_n.x.array[:] = T_init

    # ── Variational form (theta-method) ──────────────────────────────────────
    v       = ufl.TestFunction(V)
    T_trial = ufl.TrialFunction(V)

    T_theta = theta * T_trial + (1.0 - theta) * T_n

    a = (
        T_trial * v * ufl.dx
        + dt * alpha_field * ufl.dot(ufl.grad(T_theta), ufl.grad(v)) * ufl.dx
    )
    L = T_n * v * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc],
                            petsc_options={"ksp_type": "cg", "pc_type": "ilu"},
                            petsc_options_prefix="heat_")

    # ── Write diffusivity field for inspection in ParaView ───────────────────
    with io.XDMFFile(msh.comm, "results/pore_geometry.xdmf", "w") as xdmf_pores:
        xdmf_pores.write_mesh(msh)
        xdmf_pores.write_function(alpha_field)
    print(f"Diffusivity field written to results/pore_geometry.xdmf")

    # ── Time loop ─────────────────────────────────────────────────────────────
    with io.XDMFFile(msh.comm, output_file, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(T_n, 0.0)

        t = 0.0
        while t < t_end - 1e-12:
            t += dt

            T_n.x.array[:] = problem.solve().x.array

            xdmf.write_function(T_n, t)
            print(f"  t = {t:.3f}  T_max = {T_n.x.array.max():.2f} K")

    # PETSc objects are destroyed here when `problem` goes out of scope,
    # before MPI finalizes — prevents the script from hanging at exit.


run()
print("Done. Results written to", output_file)
gc.collect()      # destroy PETSc objects first
os._exit(0)       # hard exit — skips atexit/MPI finalizer that causes the hang
