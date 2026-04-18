"""
Random spherical pore/inclusion generator for 3-D unit-cube domains.

Usage
-----
from thermosim.pore_generator import PoreGenerator

gen = PoreGenerator(n_pores=20, r_min=0.03, r_max=0.08, seed=42)
gen.summary()

# In FEniCSx: build a spatially-varying diffusivity field
alpha_field = gen.make_diffusivity_field(V, alpha_matrix, alpha_pore)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Pore:
    center: np.ndarray   # shape (3,)
    radius: float


class PoreGenerator:
    """
    Places non-overlapping spherical pores randomly inside the unit cube [0,1]³.

    Parameters
    ----------
    n_pores   : number of pores to place
    r_min     : minimum pore radius
    r_max     : maximum pore radius
    margin    : minimum gap between pore surfaces (and to domain boundary)
    max_tries : rejection-sampling attempts before giving up on a pore
    seed      : random seed for reproducibility
    """

    def __init__(
        self,
        n_pores:   int   = 20,
        r_min:     float = 0.03,
        r_max:     float = 0.08,
        margin:    float = 0.005,
        max_tries: int   = 5000,
        seed:      int   = 0,
    ):
        self.r_min     = r_min
        self.r_max     = r_max
        self.margin    = margin
        self.max_tries = max_tries
        self.pores: list[Pore] = []

        rng = np.random.default_rng(seed)
        self._place(n_pores, rng)

    # ── placement ────────────────────────────────────────────────────────────

    def _place(self, n_pores: int, rng: np.random.Generator) -> None:
        placed = 0
        for _ in range(n_pores * self.max_tries):
            r = rng.uniform(self.r_min, self.r_max)
            # keep sphere fully inside the domain with margin
            lo, hi = r + self.margin, 1.0 - r - self.margin
            if lo >= hi:
                continue
            c = rng.uniform(lo, hi, size=3)

            if self._overlaps(c, r):
                continue

            self.pores.append(Pore(center=c, radius=r))
            placed += 1
            if placed == n_pores:
                break

        if placed < n_pores:
            print(
                f"[PoreGenerator] Warning: only placed {placed}/{n_pores} pores "
                f"(domain too crowded or radii too large)."
            )

    def _overlaps(self, center: np.ndarray, radius: float) -> bool:
        for p in self.pores:
            dist = np.linalg.norm(center - p.center)
            if dist < radius + p.radius + self.margin:
                return True
        return False

    # ── query ─────────────────────────────────────────────────────────────────

    def is_pore(self, x: np.ndarray) -> np.ndarray:
        """
        Vectorised membership test.

        Parameters
        ----------
        x : array of shape (3, N)  — FEniCSx coordinate layout

        Returns
        -------
        Boolean array of shape (N,); True where the point is inside any pore.
        """
        result = np.zeros(x.shape[1], dtype=bool)
        for p in self.pores:
            dx = x[0] - p.center[0]
            dy = x[1] - p.center[1]
            dz = x[2] - p.center[2]
            result |= (dx**2 + dy**2 + dz**2) <= p.radius**2
        return result

    # ── FEniCSx helper ────────────────────────────────────────────────────────

    def make_diffusivity_field(self, msh, alpha_matrix: float, alpha_pore: float):
        """
        Return a DG0 FEniCSx Function (cell-wise constant) with:
          alpha_matrix  outside pores
          alpha_pore    inside pores

        DG0 gives sharp pore boundaries because each element has exactly one
        value, evaluated at its centroid — no smearing across element edges.

        Parameters
        ----------
        msh          : dolfinx Mesh
        alpha_matrix : thermal diffusivity of the solid  [m²/s]
        alpha_pore   : thermal diffusivity inside pores  [m²/s]
                       (use air ~2e-5 or near-zero for void/insulating pores)
        """
        from dolfinx import fem
        import numpy as np

        V_dg = fem.functionspace(msh, ("DG", 0))
        alpha_f = fem.Function(V_dg, name="alpha")

        # DG0 interpolation evaluates the lambda at each cell's centroid
        alpha_f.interpolate(
            lambda x: np.where(self.is_pore(x), alpha_pore, alpha_matrix)
        )
        return alpha_f

    # ── diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> None:
        total_vol = sum((4 / 3) * np.pi * p.radius**3 for p in self.pores)
        porosity  = total_vol  # unit cube has volume 1
        print(f"Pores placed : {len(self.pores)}")
        print(f"Porosity     : {porosity * 100:.1f} %")
        print(f"Radius range : {min(p.radius for p in self.pores):.4f} – "
              f"{max(p.radius for p in self.pores):.4f}")
