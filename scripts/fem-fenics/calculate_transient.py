"""
Transient heat diffusion — anisotropic GFRP, convective cooling (θ-scheme).

PDE  : ρ·cp · dT/dt = div(K · grad(T))
IC   : T = T_init  (uniform)
BC   : -K·grad(T)·n = h·(T − T_amb)  on selected exterior faces  (Robin / convective)
         tag 1 → y_max (top face)     h = h_top
         tag 2 → x_min (left face)    h = h_xmin

Unit system: mm | s | tonne  (Abaqus mm-s-t)
  1 W/(m²·K) = 1e-3 t/(s³·K)
"""

import gc
import os
os.environ.setdefault("OMP_NUM_THREADS", "10")      # must be before numpy/MPI
from pathlib import Path
import numpy as np
from mpi4py import MPI

import pyvista                                              # type: ignore[import-untyped]
from dolfinx import fem, io, mesh, plot                     # type: ignore[import-untyped]
from dolfinx.fem.petsc import LinearProblem                 # type: ignore[import-untyped]
import ufl                                                  # type: ignore[import-untyped]

from time import time

# ── Parameters ────────────────────────────────────────────────────────────────
# Material: glass-fibre reinforced plastic (GFRP), transversely isotropic
k_par     = 0.85      # conductivity ∥ fibers  [t·mm/s³/K]  ≈ 0.85 W/(m·K)
k_perp    = 0.65      # conductivity ⊥ fibers  [t·mm/s³/K]  ≈ 0.65 W/(m·K)
rho       = 2.01e-9   # density                [t/mm³]       ≈ 2010 kg/m³
cp        = 9.02e8    # specific heat          [mm²/(s²·K)]  ≈ 902  J/(kg·K)
fiber_dir = np.array([0.0, 0.0, 1.0])

# Time integration — θ-scheme (unconditionally stable for θ ≥ 0.5)
#   θ = 1.0  → backward Euler   (1st order, most diffusive)
#   θ = 0.5  → Crank-Nicolson   (2nd order, least diffusive)
#   θ = 0.6  → recommended for coarse meshes (2nd order + light damping)
t_end  = 1000.0          # simulation end time  [s]
dt     = 10.0         # time step            [s]
theta  = 0.6          # time-integration parameter

show_plots = True     # False: skip all pyvista windows (useful for headless/parallel runs)

# Thermal boundary conditions
T_init  = 50.0        # initial uniform temperature  [°C]
T_amb   = 20.0        # ambient temperature           [°C]

h_top   = 0.001         # Robin h on top face  (y_max)  [t/(s³·K)]  = 1000 W/(m²·K)


mesh_file = "./mesh/ref_01_5mm.xdmf"

_stem = Path(mesh_file).stem
_out_dir = Path("output")
_comm = MPI.COMM_WORLD

if _comm.rank == 0:
    _out_dir.mkdir(exist_ok=True)
    _existing = sorted(_out_dir.glob(f"{_stem}_*.xdmf"))
    _next_idx = int(_existing[-1].stem.rsplit("_", 1)[-1]) + 1 if _existing else 1
    output_file = str(_out_dir / f"{_stem}_{_next_idx:03d}.xdmf")
else:
    output_file = None

output_file = _comm.bcast(output_file, root=0)


def run() -> None:
    if _comm.rank == 0:
        t0 = time()
    # ── 1. Load mesh ───────────────────────────────────────────────
    with io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")

    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, tdim)
    msh.topology.create_connectivity(tdim, fdim)

    n_elements = msh.topology.index_map(tdim).size_global
    vol = fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx))

    if msh.comm.rank == 0:
        print(
            f"Mesh   : {mesh_file}  |  {n_elements} elements  |  "
            f"volume {vol:.2f} mm³  |  {msh.topology.cell_type}\n"
            f"Solver : t_end={t_end:.1f}s  dt={dt:.4f}s  "
            f"steps={int(t_end/dt)}  θ={theta}  ranks={msh.comm.size}"
        )
    # ── 2. BC surface geometry ────────────────────────────────────────────────
    coords  = msh.geometry.x
    y_max   = msh.comm.allreduce(coords[:, 1].max(), op=MPI.MAX)
    tol     = 1e-10

    n_unit = fiber_dir / np.linalg.norm(fiber_dir)

    # ── 3. Mesh preview (rank 0 only — skipped in parallel runs) ─────────────
    if msh.comm.size == 1 and show_plots:

        topology, cell_types, geometry = plot.vtk_mesh(msh, tdim)
        vol_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        bdry_top, bdry_ct, bdry_geo = plot.vtk_mesh(msh, fdim)
        bdry_grid = pyvista.UnstructuredGrid(bdry_top, bdry_ct, bdry_geo)

        cell_centers = bdry_grid.cell_centers().points
        top_surf  = bdry_grid.extract_cells(
            np.where(np.isclose(cell_centers[:, 1], y_max, atol=tol))[0]
        )
        

        arrow_len = vol_grid.length / 10
        centers   = vol_grid.cell_centers().points[::10]
        directions = np.tile(n_unit * arrow_len, (len(centers), 1))

        pl = pyvista.Plotter(shape=(1, 2), title="Mesh preview")

        pl.subplot(0, 0)
        pl.add_text("Volume + fiber direction", font_size=10)
        pl.add_mesh(vol_grid, show_edges=True, color="lightgrey", opacity=0.4)
        pl.add_arrows(centers, directions, color="crimson", label="fiber dir")
        pl.add_legend()
        pl.add_axes()

        pl.subplot(0, 1)
        pl.add_text(f"BC: top y={y_max:.2f}  h={h_top}", font_size=10)
        pl.add_mesh(vol_grid, color="lightgrey", opacity=0.15)
        pl.add_mesh(top_surf, show_edges=True, color="tomato", opacity=1.0)
        pl.add_axes()


        pl.link_views()
        pl.show()

    # ── 4. Function space & initial condition ─────────────────────────────────
    V    = fem.functionspace(msh, ("Lagrange", 1))   # P2 for solution (mesh degree = 1)
    V_p1 = fem.functionspace(msh, ("Lagrange", 1))   # P1 for XDMF output (mesh degree = 1)
    T_n  = fem.Function(V,    name="T")
    T_out = fem.Function(V_p1, name="T")
    T_n.x.array[:] = T_init

    # ── 5. Material tensors ───────────────────────────────────────────────────
    # K = k_perp·I + (k_par−k_perp)·(n⊗n)  — transversely isotropic
    n_ufl  = ufl.as_vector(n_unit)
    K      = k_perp * ufl.Identity(3) + (k_par - k_perp) * ufl.outer(n_ufl, n_ufl)
    # K_rcp  = (1.0 / (rho * cp)) * K             # thermal diffusivity tensor  [mm²/s]
    # h_rcp_top  = h_top  / (rho * cp)            # h_top  / (ρ·cp)             [mm/s]
    

    # ── 6. Boundary measures — two tagged surfaces ────────────────────────────
    # tag 1: y_max  (top face)
    top_facets = np.sort(mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], y_max, atol=tol)
    ))


    print(f"Top facets  (y={y_max:.4f}): {len(top_facets)}")


    all_facets = np.concatenate([top_facets])
    all_tags   = np.concatenate([
        np.ones(len(top_facets),  dtype=np.int32),
    ])
    order = np.argsort(all_facets)
    mt = mesh.meshtags(msh, fdim, all_facets[order], all_tags[order])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=mt)

    area_top  = fem.assemble_scalar(fem.form(1.0 * ds(1)))
    print(f"Top area: {area_top:.4f}  ")

    # ── 7. Variational form (θ-scheme) ────────────────────────────────────────
    # Weak form of ρ·cp·dT/dt = div(K·grad(T)) with Robin BC h·(T−T_amb) on Γ:
    #   LHS (implicit, θ):   ρ·cp·∫T·v dx + θ·dt·∫(K·∇T)·∇v dx + θ·dt·h·∫T·v ds
    #   RHS (explicit, 1−θ): ρ·cp·∫T_n·v dx − (1−θ)·dt·∫(K·∇T_n)·∇v dx
    #                          − (1−θ)·dt·h·∫T_n·v ds + dt·h·T_amb·∫v ds
    #   θ=1 → backward Euler,  θ=0.5 → Crank-Nicolson,  θ=0.6 → recommended
    v       = ufl.TestFunction(V)
    T_trial = ufl.TrialFunction(V)

    # Bind measures to the mesh so domain is never lost when scalar coefficients are zero
    dx = ufl.dx(domain=msh)

    diff_new = ufl.dot(ufl.dot(K, ufl.grad(T_trial)), ufl.grad(v))
    diff_old = ufl.dot(ufl.dot(K, ufl.grad(T_n)),     ufl.grad(v))

    a = (
        rho*cp * T_trial * v * dx
        + theta * dt * diff_new * dx
        + theta * dt * h_top * T_trial * v * ds(1)
    )
    L = (  # type: ignore[operator]
        rho*cp * T_n * v * dx
        - (1 - theta) * dt * diff_old * dx                # type: ignore[operator]
        - (1 - theta) * dt * h_top * T_n * v * ds(1)      # type: ignore[operator]
        + dt * h_top * T_amb * v * ds(1)                   # type: ignore[operator]
    )



    t_compile = time()
    problem = LinearProblem(a, L, bcs=[],
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                           "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="heat_"
                            )
    if msh.comm.rank == 0:
        print(f"Form compilation + LU factorisation: {time() - t_compile:.2f}s")

    # ── 8. Time loop ──────────────────────────────────────────────────────────
    snapshots: list[tuple[float, np.ndarray]] = [(0.0, T_n.x.array.copy())]  # type: ignore[union-attr]
    t_solve_total = 0.0

    with io.XDMFFile(msh.comm, output_file, "w") as xdmf:
        xdmf.write_mesh(msh)
        T_out.interpolate(T_n)
        xdmf.write_function(T_out, 0.0)  # type: ignore[arg-type]

        t = 0.0
        while t < t_end - 1e-12:
            t += dt
            t_step = time()
            T_n.x.array[:] = problem.solve().x.array  # type: ignore[union-attr]
            t_solve_total += time() - t_step
            T_out.interpolate(T_n)
            xdmf.write_function(T_out, t)             # type: ignore[arg-type]
            snapshots.append((t, T_n.x.array.copy()))  # type: ignore[union-attr]
            T_max = msh.comm.allreduce(float(T_n.x.array.max()), op=MPI.MAX)
            T_min = msh.comm.allreduce(float(T_n.x.array.min()), op=MPI.MIN)
            if msh.comm.rank == 0:
                print(f"  t={t:.4f}s \t T_max={T_max:.3f} \t T_min={T_min:.3f}")
   
    # ── 9. Interactive result viewer (rank 0, single-process only) ───────────
    if msh.comm.size == 1 and show_plots:
        topology, cell_types, geometry = plot.vtk_mesh(V)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        all_T        = np.array([s[1] for s in snapshots])
        all_t        = [s[0] for s in snapshots]
        T_global_min = float(all_T.min())
        T_global_max = float(all_T.max())

        grid.point_data["T"] = all_T[0]
        grid.set_active_scalars("T")

        pl = pyvista.Plotter(title="Temperature — drag slider to scrub time")
        pl.add_mesh(grid, scalars="T", cmap="coolwarm",
                    clim=[T_global_min, T_global_max],
                    scalar_bar_args={"title": "Temperature [°C]"})

        def update_time(value: float) -> None:
            idx = int(round(value))
            grid.point_data["T"] = all_T[idx]
            pl.add_text(f"$t$ = {all_t[idx]:.4f} s", name="time_label",
                        position="upper_left", font_size=12)

        pl.add_slider_widget(update_time,
                             rng=[0, len(snapshots) - 1], value=0,
                             title="Time step",
                             pointa=(0.3, 0.9), pointb=(0.9, 0.9),
                             style="modern")
        update_time(0)
        pl.add_axes()
        pl.show()

    if msh.comm.rank == 0:
        t1 = time()
        n_steps = int(t_end / dt)
        print(
            f"Solve loop : {t_solve_total:.2f}s total  "
            f"({t_solve_total/n_steps*1000:.1f} ms/step)\n"
            f"Total      : {t1 - t0:.2f}s  →  {output_file}"
        )

run()

gc.collect()
os._exit(0)
