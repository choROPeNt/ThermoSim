# Thermal Mapping

Maps the undistorted IR temperature field onto the FEM surface mesh so that measured boundary conditions can be compared to or imposed on the simulation.

---

## Pipeline overview

```
IR frames (.csv / .irb)
        ↓
  load_irb_txt()
        ↓
  undistort (cv2)          ← K, dist from ir_calib.npz
        ↓
  scale pixels → mm        ← Z / f_x,  Z / f_y
        ↓
  export VTU (meshio)      →  undistorted_thermal.vtu
        ↓
  load FEM mesh (.xdmf)
        ↓
  extract surface nodes (y ≈ 0)
        ↓
  RegularGridInterpolator
        ↓
  export mapped VTU        →  fem_thermal_mapped.vtu
```

---

## Notebooks

### `data_io.ipynb`

Loads IRB/CSV frames, applies undistortion, and exports the thermal image as a quad VTU mesh in millimetres.

Key cell — VTU export:

```python
calib  = np.load("../data/ir_calib.npz")
K      = calib["K"]
dist   = calib["dist"].flatten()

Z_mm   = 500.0               # camera-to-surface distance
dx_mm  = Z_mm / K[0, 0]     # mm per pixel in x
dy_mm  = Z_mm / K[1, 1]     # mm per pixel in y
```

Output: `results/undistorted_thermal.vtu` with point data field `temperature` (°C).

---

### `thermal_mapping.ipynb`

Loads the VTU and the FEM mesh, aligns coordinate systems, and resamples.

```python
from scipy.interpolate import RegularGridInterpolator

interp = RegularGridInterpolator((ty, tx), T_img, method="linear",
                                  bounds_error=False, fill_value=np.nan)
T_surface = interp(np.column_stack([y_q, x_q]))
```

**Transform parameters** — adjust to your setup:

| Variable | Meaning |
|----------|---------|
| `FACE_Y` | y-coordinate of the FEM face visible to the camera |
| `scale_x`, `scale_y` | Ratio of thermal extent to FEM extent |
| `offset_x`, `offset_y` | Manual shift to align image to mesh (mm) |

Output: `results/fem_thermal_mapped.vtu` with point field `temperature_ir` (NaN on interior nodes).

---

## Viewing in ParaView

1. Open `fem_thermal_mapped.vtu`
2. Color by `temperature_ir`
3. Apply **Threshold** filter to hide NaN interior nodes if needed

To overlay simulation and measurement:

1. Open both `temperature.xdmf` (FEM result) and `fem_thermal_mapped.vtu`
2. Add **Calculator** filter: `temperature_sim - temperature_ir` → residual field
