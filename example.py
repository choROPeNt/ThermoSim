"""
Transient heat conduction using FEniCSx (DOLFINx).

Solves:  rho*cp * dT/dt - div( K · grad(T) ) = Q   on [0,Lx] x [0,Ly]

Boundary conditions:
  left  : T = T_hot   (heat source side)
  right : T = T_cold  (heat sink side)
  top / bottom : insulated (natural BC — zero flux)

Initial condition: T = T_cold everywhere

Material: anisotropic GFRP
  k_x = conductivity along fibers  (x-direction)
  k_y = conductivity transverse

Heat source Q: Gaussian hot spot at centre (e.g. embedded resistor)
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

# ── Parameters ────────────────────────────────────────────────────────────────
nx, ny  = 64, 64      # mesh resolution

# material — GFRP (anisotropic)
k_x    = 1.0          # along-fiber conductivity   [W/(m·K)]
k_y    = 0.35         # transverse conductivity     [W/(m·K)]
rho    = 1850.0       # density                     [kg/m³]
cp     = 1200.0       # specific heat               [J/(kg·K)]

# geometry
Lx, Ly = 1.0, 1.0    # plate dimensions [m]

# temperatures
T_hot  = 100.0        # left edge   [°C]
T_cold =  20.0        # right edge  [°C]

# heat source — Gaussian hot spot at centre
Q_peak = 5000.0       # peak heat generation  [W/m³]
sigma  = 0.05         # spot radius           [m]

# time
t_end  = 3600.0       # end time   [s]
dt     = 60.0         # time step  [s]
theta  = 1.0          # 1 = backward Euler, 0.5 = Crank-Nicolson

output_file = "results/temperature_transient.xdmf"


def run():
    # ── Mesh & function space ─────────────────────────────────────────────────
    msh = dmesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0.0, 0.0], [Lx, Ly]],
        [nx, ny],
        cell_type=dmesh.CellType.quadrilateral,
    )
    V = fem.functionspace(msh, ("Lagrange", 1))

    # ── Boundary conditions ───────────────────────────────────────────────────
    bc_left = fem.dirichletbc(
        T_hot,
        fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0)),
        V,
    )
    bc_right = fem.dirichletbc(
        T_cold,
        fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], Lx)),
        V,
    )

    # ── Anisotropic conductivity tensor ───────────────────────────────────────
    K = ufl.as_tensor([[k_x, 0.0],
                       [0.0, k_y]])

    # ── Heat source — Gaussian hot spot ───────────────────────────────────────
    x  = ufl.SpatialCoordinate(msh)
    r2 = (x[0] - 0.5 * Lx)**2 + (x[1] - 0.5 * Ly)**2
    Q  = Q_peak * ufl.exp(-r2 / (2 * sigma**2))

    # ── Initial condition ─────────────────────────────────────────────────────
    T_n = fem.Function(V, name="T")   # solution at previous step
    T_n.x.array[:] = T_cold

    # ── Variational form (theta-method) ───────────────────────────────────────
    T_trial = ufl.TrialFunction(V)
    v       = ufl.TestFunction(V)
    T_theta = theta * T_trial + (1.0 - theta) * T_n

    a = (
        rho * cp * T_trial * v * ufl.dx
        + dt * ufl.dot(K * ufl.grad(T_theta), ufl.grad(v)) * ufl.dx
    )
    L = (
        rho * cp * T_n * v * ufl.dx
        + dt * Q * v * ufl.dx
    )

    problem = LinearProblem(a, L, bcs=[bc_left, bc_right],
                            petsc_options={"ksp_type": "cg", "pc_type": "ilu"},
                            petsc_options_prefix="transient_heat_")

    print(f"  Material : GFRP  k_x={k_x} W/(m·K)  k_y={k_y} W/(m·K)  "
          f"rho={rho} kg/m³  cp={cp} J/(kg·K)")
    print(f"  Anisotropy ratio k_x/k_y : {k_x/k_y:.1f}x")

    # ── Time loop ─────────────────────────────────────────────────────────────
    with io.XDMFFile(msh.comm, output_file, "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_function(T_n, 0.0)

        t = 0.0
        while t < t_end - 1e-12:
            t += dt
            T_n.x.array[:] = problem.solve().x.array
            xdmf.write_function(T_n, t)
            print(f"  t = {t:.0f} s  T_max = {T_n.x.array.max():.2f} °C"
                  f"  T_min = {T_n.x.array.min():.2f} °C"
                  f"  T_mean = {T_n.x.array.mean():.2f} °C")

    # PETSc objects destroyed here before MPI shuts down


run()
print(f"\nDone. Results written to {output_file}")
gc.collect()
os._exit(0)
