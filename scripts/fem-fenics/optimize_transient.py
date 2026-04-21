"""
Inverse calibration — 3-D plate cooling.

The analytical solution acts as a placeholder for experimental measurements.
The optimizer recovers k (conductivity) and h (convection coefficient) by
minimizing MSE between the FEM forward model and the "measured" surface temperature.

Swap  T_meas = [T_analytical(L, t) for t in t_meas]
with  T_meas = load_experimental_data(...)   to use real data.
"""
import os, gc
import numpy as np
from scipy.optimize import brentq, minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

# ── Fixed geometry & material (known) ────────────────────────────────────────
W      = 0.10     # plate width              [m]
H      = 0.08     # plate height             [m]
L      = 0.02     # plate thickness          [m]  ← cooling direction z
rho    = 2010.0   # density                  [kg/m³]
cp     = 902.0    # specific heat            [J/(kg·K)]
T_init = 50.0     # initial temperature      [°C]
T_amb  = 20.0     # ambient temperature      [°C]

# ── True parameters (used ONLY to generate synthetic measurements) ────────────
k_true = 0.85     # conductivity             [W/(m·K)]
h_true = 50.0     # convection coefficient   [W/(m²·K)]

# ── Optimiser starting guess (deliberately perturbed) ────────────────────────
k_init = 1.30     # initial guess for k
h_init = 80.0     # initial guess for h

# ── Time settings ─────────────────────────────────────────────────────────────
t_end  = 300.0
dt     = 5.0
theta  = 0.6

# ── Noise on measurements (0 = perfect; >0 simulates sensor noise) ────────────
noise_std = 0.05   # [°C]

# ── Mesh ──────────────────────────────────────────────────────────────────────
nx, ny, nz = 4, 4, 40
rng = np.random.default_rng(0)

# ── Analytical solution (placeholder for experimental data) ──────────────────
def _eigenvalues(Bi: float, n: int = 60) -> np.ndarray:
    roots = []
    for i in range(n):
        a, b = i*np.pi + 1e-10, i*np.pi + np.pi/2 - 1e-10
        try:
            roots.append(brentq(lambda x: x*np.tan(x) - Bi, a, b))
        except ValueError:
            pass
    return np.array(roots)

def T_surface_analytical(t: float, k: float, h: float) -> float:
    """Analytical surface temperature T(z=L, t) for given k, h."""
    if t == 0.0:
        return T_init
    alpha  = k / (rho * cp)
    Bi     = h * L / k
    lams   = _eigenvalues(Bi)
    Fo     = alpha * t / L**2
    s = sum(
        4*np.sin(lam)/(2*lam + np.sin(2*lam)) * np.cos(lam) * np.exp(-lam**2 * Fo)
        for lam in lams
    )
    return T_amb + (T_init - T_amb) * s

# ── Synthetic measurements ────────────────────────────────────────────────────
# ↓↓ Replace this block with real experimental data ↓↓
t_meas = np.arange(dt, t_end + dt/2, dt * 3)   # every 3rd time step
T_meas = np.array([T_surface_analytical(t, k_true, h_true) for t in t_meas])
T_meas += rng.normal(0, noise_std, len(T_meas))  # add sensor noise
# ↑↑ ─────────────────────────────────────────────────────────────────────── ↑↑

print(f"Measurements: {len(t_meas)} points  "
      f"T range [{T_meas.min():.2f}, {T_meas.max():.2f}] °C")

# ── Build mesh once ───────────────────────────────────────────────────────────
msh  = dmesh.create_box(MPI.COMM_WORLD,
                        [[0.0, 0.0, 0.0], [W, H, L]],
                        [nx, ny, nz],
                        dmesh.CellType.tetrahedron)
V    = fem.functionspace(msh, ("Lagrange", 1))
fdim = msh.topology.dim - 1

top_facets = dmesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[2], L))
mt = dmesh.meshtags(msh, fdim, np.sort(top_facets),
                    np.ones(len(top_facets), dtype=np.int32))
ds   = ufl.Measure("ds", domain=msh, subdomain_data=mt)
dx   = ufl.dx(domain=msh)
_bbt = bb_tree(msh, msh.topology.dim)

n_dof = V.dofmap.index_map.size_global
print(f"Mesh: {nx}×{ny}×{nz}  |  {msh.topology.index_map(3).size_global} cells  |  {n_dof} DOFs")

# ── Point evaluation ──────────────────────────────────────────────────────────
_surf_pt = np.array([[W/2, H/2, L]])

def _eval_pt(T_h, pt):
    cand  = compute_collisions_points(_bbt, pt)
    cells = compute_colliding_cells(msh, cand, pt).array[
                compute_colliding_cells(msh, cand, pt).offsets[:-1]]
    return float(T_h.eval(pt, cells).flatten()[0])

# ── FEM forward model — compile forms ONCE with fem.Constant ─────────────────
# Using fem.Constant means FFCx compiles the form a single time;
# subsequent calls just update the constant values and reassemble.
_k_c = fem.Constant(msh, float(k_init))
_h_c = fem.Constant(msh, float(h_init))
_T_n = fem.Function(V, name="T")

_v, _T_tr = ufl.TestFunction(V), ufl.TrialFunction(V)
_diff_new = _k_c * ufl.dot(ufl.grad(_T_tr), ufl.grad(_v))
_diff_old = _k_c * ufl.dot(ufl.grad(_T_n),  ufl.grad(_v))

_a = (rho*cp * _T_tr * _v * dx
      + theta * dt * _diff_new * dx
      + theta * dt * _h_c * _T_tr * _v * ds(1))
_Lf = (rho*cp * _T_n * _v * dx
       - (1-theta) * dt * _diff_old * dx
       - (1-theta) * dt * _h_c * _T_n * _v * ds(1)
       + dt * _h_c * T_amb * _v * ds(1))

_prob = LinearProblem(_a, _Lf, bcs=[],
                      petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                      petsc_options_prefix="opt_")

eval_log: list[tuple] = []   # (k, h, mse) for convergence plot

def forward(k_val: float, h_val: float) -> np.ndarray:
    """Run transient FEM, return surface T at t_meas times."""
    _k_c.value = k_val
    _h_c.value = h_val
    _T_n.x.array[:] = T_init

    T_out = []
    t = 0.0
    while t < t_end - 1e-10:
        t += dt
        _T_n.x.array[:] = _prob.solve().x.array
        if any(np.isclose(t, t_meas)):
            T_out.append(_eval_pt(_T_n, _surf_pt))

    return np.array(T_out)


def objective(params: np.ndarray) -> float:
    k_val, h_val = float(params[0]), float(params[1])
    if k_val <= 0 or h_val <= 0:
        return 1e10
    T_pred = forward(k_val, h_val)
    mse = float(np.mean((T_pred - T_meas)**2))
    eval_log.append((k_val, h_val, mse))
    print(f"  k={k_val:.4f}  h={h_val:.2f}  MSE={mse:.4e}")
    return mse

# ── Model / mesh preview ──────────────────────────────────────────────────────
def _box_faces(W, H, L):
    v = np.array([[0,0,0],[W,0,0],[W,H,0],[0,H,0],
                  [0,0,L],[W,0,L],[W,H,L],[0,H,L]]) * 1e3
    return [[v[0],v[1],v[2],v[3]], [v[4],v[5],v[6],v[7]],
            [v[0],v[1],v[5],v[4]], [v[2],v[3],v[7],v[6]],
            [v[0],v[3],v[7],v[4]], [v[1],v[2],v[6],v[5]]], v

fig_pre = plt.figure(figsize=(12, 4))

ax3 = fig_pre.add_subplot(131, projection="3d")
faces, _ = _box_faces(W, H, L)
colors = ["lightgrey"]*6; colors[1] = "tomato"
ax3.add_collection3d(Poly3DCollection(faces, facecolors=colors,
                                      edgecolors="grey", linewidths=0.5, alpha=0.5))
ax3.set_xlim(0, W*1e3); ax3.set_ylim(0, H*1e3); ax3.set_zlim(0, L*1e3)
ax3.set_xlabel("x [mm]"); ax3.set_ylabel("y [mm]"); ax3.set_zlabel("z [mm]")
ax3.set_title(f"{W*1e3:.0f}×{H*1e3:.0f}×{L*1e3:.0f} mm\nRed=Robin  Grey=Adiabatic", fontsize=9)

ax2 = fig_pre.add_subplot(132)
ax2.barh(0, L*1e3, height=6, color="lightsteelblue", edgecolor="steelblue")
ax2.annotate("Adiabatic", xy=(0, 0), xytext=(-2, 2), ha="right", fontsize=9,
             color="grey", arrowprops=dict(arrowstyle="->", color="grey"))
ax2.annotate(f"Robin h={h_true} W/(m²K)", xy=(L*1e3, 0), xytext=(L*1e3+1, 2),
             ha="left", fontsize=9, color="tomato",
             arrowprops=dict(arrowstyle="->", color="tomato"))
ax2.set_xlabel("z [mm]"); ax2.set_yticks([])
ax2.set_title(f"BCs  |  ρ={rho}  cp={cp}\nθ={theta}  dt={dt}s  nz={nz}", fontsize=9)
ax2.grid(axis="x", alpha=0.3)

ax_m = fig_pre.add_subplot(133)
ax_m.plot(t_meas, T_meas, "o-", color="steelblue", ms=5, label="Measurements (+ noise)")
ax_m.axhline(T_amb, ls=":", color="grey", label=f"T_amb={T_amb}°C")
ax_m.set_xlabel("t [s]"); ax_m.set_ylabel("T_surface [°C]")
ax_m.set_title(f"Synthetic measurements\n"
               f"k_true={k_true}  h_true={h_true}  noise={noise_std}°C", fontsize=9)
ax_m.legend(fontsize=8); ax_m.grid(True)

plt.tight_layout()
plt.savefig("output/plate_model_preview.png", dpi=150)
plt.show()

# ── Run optimisation ──────────────────────────────────────────────────────────
print(f"\nStarting optimisation  x0=[k={k_init}, h={h_init}]")
result = minimize(
    objective,
    x0=[k_init, h_init],
    method="Nelder-Mead",
    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 200, "adaptive": True},
)

k_opt, h_opt = result.x
print(f"\n{'─'*50}")
print(f"True   : k={k_true}   h={h_true}")
print(f"Init   : k={k_init}   h={h_init}")
print(f"Found  : k={k_opt:.5f}   h={h_opt:.3f}")
print(f"Error  : Δk={abs(k_opt-k_true):.2e}   Δh={abs(h_opt-h_true):.2e}")
print(f"Evals  : {len(eval_log)}")

# ── Results plot ──────────────────────────────────────────────────────────────
ks   = [e[0] for e in eval_log]
hs   = [e[1] for e in eval_log]
mses = [e[2] for e in eval_log]

T_init_pred = forward(k_init, h_init)
T_opt_pred  = forward(k_opt,  h_opt)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Panel 1 — surface temperature comparison
ax = axes[0]
ax.plot(t_meas, T_meas,       "o",  color="steelblue", ms=6, label="Measurements", zorder=5)
ax.plot(t_meas, T_init_pred,  "--", color="orange",    lw=2, label=f"FEM initial (k={k_init}, h={h_init})")
ax.plot(t_meas, T_opt_pred,   "-",  color="tomato",    lw=2, label=f"FEM optimised (k={k_opt:.3f}, h={h_opt:.1f})")
ax.set_xlabel("t [s]"); ax.set_ylabel("T_surface [°C]")
ax.set_title("Surface temperature — before & after"); ax.legend(fontsize=8); ax.grid(True)

# Panel 2 — parameter search path
ax = axes[1]
sc = ax.scatter(ks, hs, c=np.log10(np.array(mses)+1e-12),
                cmap="plasma", s=30, zorder=3)
ax.scatter([k_true], [h_true], marker="*", s=200, color="green",  zorder=5, label="True")
ax.scatter([k_init], [h_init], marker="s", s=80,  color="orange", zorder=5, label="Init")
ax.scatter([k_opt],  [h_opt],  marker="D", s=80,  color="tomato", zorder=5, label="Opt")
plt.colorbar(sc, ax=ax, label="log₁₀(MSE)")
ax.set_xlabel("k [W/(m·K)]"); ax.set_ylabel("h [W/(m²·K)]")
ax.set_title("Optimizer search path"); ax.legend(fontsize=8); ax.grid(True)

# Panel 3 — MSE convergence
ax = axes[2]
ax.semilogy(range(len(mses)), mses, lw=1.5, color="purple")
ax.set_xlabel("Evaluation #"); ax.set_ylabel("MSE [°C²]")
ax.set_title("Convergence"); ax.grid(True)

plt.tight_layout()
plt.savefig("output/plate_cooling_results.png", dpi=150)
plt.show()

gc.collect()
os._exit(0)
